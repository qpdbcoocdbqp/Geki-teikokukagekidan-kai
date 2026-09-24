"""Trained checkpoints: a run directory or a Hub repo holding a LoRA adapter, `head.pt` and the tokenizer.

This module knows the layout of `head.pt` and how a checkpoint becomes a
`DecisionModel` for the local FastAPI and quantized example servers.

    ck = Checkpoint("jaredpalmer/kev-4b")          # or a local run directory; `@tag` pins a Hub revision
    tok, model = ck.load("mps", LoadOptions.from_env())
    ck.meta.temperature                             # the calibration the checkpoint carries
"""
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch

try:
    from ..quant import normalize_quantization, quantization_from_env
except ImportError:  # Support ``python -m server.main`` from componet/kev.
    from quant import normalize_quantization, quantization_from_env
from .model import DecisionModel, load_tokenizer

def resolve_run(run):
    """Local run directory as given, or a Hub repo id like jaredpalmer/kev-4b, optionally pinned to a revision or tag
    with `@` (jaredpalmer/kev-4b@qwen3), downloaded to the HF cache. Returns a str path."""
    if os.path.isdir(run):
        return str(run)
    from huggingface_hub import snapshot_download
    repo, _, revision = str(run).partition("@")
    return snapshot_download(repo, revision=revision or None, allow_patterns=["*.json", "*.safetensors", "*.pt", "*.txt", "*.jinja"])


@dataclass
class Meta:
    """Contents of `head.pt`. Every reader gets the same defaults for fields older checkpoints did not write.
    `extra` keeps the rest of the file (training args, suite hash, init provenance, temperature fit) so a
    read-modify-write round trip loses nothing."""
    base: str
    head: dict | None = None
    base_revision: str | None = None
    lora: int = 0
    head_dim: int = 256
    option_isolation: bool = False
    special_embeddings: bool = False
    weights_dtype: str = "fp32"
    temperature: float = 1.0
    holdout: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    KNOWN = ("base", "head", "base_revision", "lora", "head_dim", "option_isolation", "special_embeddings", "weights_dtype", "temperature", "holdout")

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: d[k] for k in cls.KNOWN if k in d}, extra={k: v for k, v in d.items() if k not in cls.KNOWN})

def read_meta(run):
    return Meta.from_dict(torch.load(f"{run}/head.pt", map_location="cpu"))


@dataclass(frozen=True)
class LoadOptions:
    """How a checkpoint is turned into a model. Defaults are the exact path every reported number uses; the fields
    are the same knobs the KEV_* environment variables expose to the command-line tools (see from_env).

    dtype        None = fp32, the exact path every reported number uses (bf16 when the checkpoint was trained with a bf16
                 backbone). kev.serve defaults to bf16 on CUDA and MPS instead: half the memory, 2-4.5x lower latency on an
                 L4 (Kev-4B: 209 -> 118 ms at 101 tokens, 850 -> 189 ms at 330 tokens), probabilities within ~0.01 and
                 the same argmax on the checks run so far. KEV_DTYPE=fp32 restores the exact path when serving.
    merge        fold the LoRA into the base weights: the delta is computed from the fp32 adapter and added in fp32 with
                 one rounding to the load dtype, so a bf16 model holds exactly round(W + delta), the same bits as merging
                 an fp32 copy and casting, without the fp32 copy (Kev-9B needed 36 GB of GPU memory to load for that).
                 Identical because the Qwen bases are stored in bf16; a base stored in fp32 would be rounded twice.
                 Exact in fp32; in bf16 it is faster (~15%) and closer to the fp32 numbers than the unmerged adapter
                 (kev-4b, 24 dev records: max |dp| 0.017 vs 0.029, 0 vs 1 argmax flips). Ignored for adapters that carry
                 trained token embeddings.
    attn         attention backend; None = the model default (SDPA on CUDA, eager elsewhere). "sdpa" on MPS measured
                 parity with eager and is a few percent faster.
    lora_scale   WiSE-FT-style interpolation between base (0) and fine-tuned weights (1), at inference.
    temperature  None = the temperature the checkpoint carries (fitted by scripts/calibrate_checkpoint.py); 1.0 = raw logits.
    cuda_graphs  replay the serving passes (state prefix, question rows on a cached state) of a hybrid backbone on CUDA as
                 CUDA graphs (kev.cuda_graphs). None = off, the eager path every reported number uses; kev.serve turns it on
                 for CUDA. Exact up to floating-point reassociation, not bit for bit (the passes are padded to buckets).
    """
    dtype: torch.dtype | None = None
    merge: bool = True
    attn: str | None = None
    lora_scale: float = 1.0
    temperature: float | None = None
    cuda_graphs: bool | None = None
    quantization: str | None = None

    @classmethod
    def from_env(cls, env=os.environ):
        """KEV_DTYPE=bf16|fp16|fp32, KEV_MERGE=0, KEV_ATTN=sdpa|eager, KEV_LORA_SCALE, KEV_TEMPERATURE,
        KEV_CUDA_GRAPHS=0|1, KEV_QUANTIZATION=none|8bit|4bit.
        For command-line entry points only; library code passes an explicit LoadOptions. Explicit values that equal a
        library default are kept (fp32 as torch.float32, "torch" as a string) so a caller with its own default, like
        kev.serve, can tell "asked for it" from "did not say"."""
        return cls(dtype={"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(env.get("KEV_DTYPE", "")),
                   merge=env.get("KEV_MERGE", "1") != "0", attn=env.get("KEV_ATTN") or None,
                   lora_scale=float(env.get("KEV_LORA_SCALE", "1")),
                   temperature=float(env["KEV_TEMPERATURE"]) if env.get("KEV_TEMPERATURE") else None,
                   cuda_graphs={"0": False, "1": True}.get(env.get("KEV_CUDA_GRAPHS", "")),
                   quantization=quantization_from_env(env))


class Checkpoint:
    def __init__(self, run):
        self.requested = str(run)                    # what the caller asked for (a Hub id stays a Hub id in labels)
        self.path = resolve_run(run)
        self.meta = read_meta(self.path)

    def file(self, name):
        return Path(self.path) / name

    def adapter_config(self):
        return json.loads(self.file("adapter_config.json").read_text(encoding="utf-8"))

    def load(self, device, opts=LoadOptions()):
        """-> (tokenizer, model) in eval mode with the LoRA and pointer head loaded."""
        meta = self.meta
        tok = load_tokenizer(meta.base, revision=meta.base_revision)
        m = self._load_torch(tok, device, opts)
        m.head.load_state_dict(meta.head); m.eval()
        m.head.temperature = meta.temperature if opts.temperature is None else opts.temperature
        return tok, m

    def _load_torch(self, tok, device, opts):
        from peft import PeftModel
        meta = self.meta
        quantization = normalize_quantization(opts.quantization)
        dtype, merge = opts.dtype or torch.float32, opts.merge
        if meta.weights_dtype == "bf16":
            # trained with a bf16 backbone (--weights_dtype bf16, e.g. the 35B-A3B MoE whose fused experts need bf16): load it
            # the same way and keep the fp32 adapter unmerged rather than folding it into bf16 weights.
            dtype, merge = torch.bfloat16, False
        merge = merge and not quantization and not self.adapter_config().get("trainable_token_indices")   # quantized and token-trained adapters stay unmerged
        m = DecisionModel(meta.base, tok, device, lora=None, revision=meta.base_revision, head_dim=meta.head_dim,
                          option_isolation=meta.option_isolation, dtype=dtype, attn=opts.attn,
                          quantization=quantization)
        m.lm = PeftModel.from_pretrained(m.lm, self.path, torch_device=str(device))
        if not quantization: m.lm = m.lm.to(device)   # quantized weights were placed by device_map and must not be moved
        if opts.lora_scale != 1:
            for module in m.lm.modules():
                if isinstance(getattr(module, "scaling", None), dict):
                    for k in module.scaling: module.scaling[k] *= opts.lora_scale
            m.lora_scale = opts.lora_scale
        if merge: m.lm = m.lm.merge_and_unload()     # W += delta: fp32 math, one rounding (see LoadOptions.merge)
        if not quantization and dtype != torch.float32: m.lm = m.lm.to(dtype)
        if opts.cuda_graphs and str(device).startswith("cuda") and m.hybrid:
            from .cuda_graphs import CudaGraphs
            m.graphs = CudaGraphs(m.lm, m.pad_id)
        return m

