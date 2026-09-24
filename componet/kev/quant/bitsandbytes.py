"""bitsandbytes configuration for loading Kev's Qwen base model.

The base is quantized while ``transformers`` loads it. Kev's LoRA adapter
remains separate, which avoids merging a floating-point delta back into
quantized weights and introducing another quantization round trip.
"""

import os
from collections.abc import Mapping
from typing import Any

import torch

QUANTIZATION_ENV = "KEV_QUANTIZATION"
_ALIASES = {
    "": None,
    "none": None,
    "off": None,
    "8bit": "8bit",
    "int8": "8bit",
    "4bit": "4bit",
    "int4": "4bit",
}


def normalize_quantization(value: str | None) -> str | None:
    """Return the canonical quantization mode or reject an invalid value."""
    key = (value or "").strip().lower()
    try:
        return _ALIASES[key]
    except KeyError as error:
        choices = "none, 8bit, or 4bit"
        raise ValueError(f"{QUANTIZATION_ENV} must be {choices}; got {value!r}") from error


def quantization_from_env(env: Mapping[str, str] = os.environ) -> str | None:
    """Read and validate ``KEV_QUANTIZATION``."""
    return normalize_quantization(env.get(QUANTIZATION_ENV))


def build_model_load_kwargs(
    mode: str | None,
    device: str,
    dtype: torch.dtype,
) -> dict[str, Any]:
    """Build ``from_pretrained`` kwargs for on-the-fly base quantization.

    ``device_map`` places the model during loading; callers must not move or
    cast the resulting quantized backbone afterwards. Non-linear layers use
    ``dtype`` while 4-bit matrix multiplication uses bf16 when possible.
    """
    mode = normalize_quantization(mode)
    if mode is None:
        return {}
    if not isinstance(device, (str, torch.device)):
        raise TypeError(
            f"device must be a string or torch.device, not {type(device).__name__}"
        )
    if not isinstance(dtype, torch.dtype):
        hint = "; use torch.bfloat16, torch.float16, or torch.float32"
        raise TypeError(f"dtype must be a torch.dtype, not {type(dtype).__name__}{hint}")
    if str(device).startswith("mps"):
        raise ValueError(
            "bitsandbytes quantization does not support MPS; use Kev's MLX "
            "backend or disable KEV_QUANTIZATION"
        )

    try:
        from transformers import BitsAndBytesConfig
    except ImportError as error:
        raise RuntimeError(
            "KEV_QUANTIZATION requires transformers with bitsandbytes support"
        ) from error

    if mode == "8bit":
        config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        compute_dtype = dtype if dtype in (torch.bfloat16, torch.float16) else torch.float32
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    return {
        "device_map": {"": str(device)},
        "quantization_config": config,
    }
