"""
PyTorch / CUDA / CPU Inference Engine for Parallel Constrained Decoding.
Optimized for Linux containers, Hugging Face Spaces (ZeroGPU & CUDA), and cloud environments.
"""

import os
import time
import json
import copy
import threading
from typing import Dict, Any, Generator

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from core.chat import tokenize_chat_prefill
from core.schema import StructuredSchema, score_full_sequences
from core.prompt_builder import build_naive_chat, build_parallel_chat

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")

_torch_model = None
_torch_tokenizer = None
_torch_device = None
_gpu_lock = threading.Lock()


def get_torch_engine():
    global _torch_model, _torch_tokenizer, _torch_device
    if _torch_model is None or _torch_tokenizer is None:
        _torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if _torch_device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32

        print(f"Loading {MODEL_ID} on {_torch_device} ({dtype})...")
        t0 = time.perf_counter()
        _torch_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        
        load_kwargs = {
            "dtype": "auto",
            "low_cpu_mem_usage": True
        }
        if _torch_device == "cuda":
            load_kwargs["device_map"] = "auto"

        _torch_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs)
        if _torch_device == "cpu":
            _torch_model = _torch_model.to("cpu")
        _torch_model.eval()
        print(f"Engine loaded on {_torch_device} in {time.perf_counter() - t0:.2f}s.")
        
    return _torch_model, _torch_tokenizer, _torch_device


def run_parallel_generation_torch(
    context: str,
    schema: StructuredSchema,
    temperature: float = 1.0
) -> Dict[str, Any]:
    """
    Parallel Constrained Decision Engine running on PyTorch (CUDA / CPU).
    Evaluates all schema fields concurrently against a broadcast prefix KV-cache.
    """
    model, tokenizer, device = get_torch_engine()
    t0 = time.perf_counter()

    # 1. Compile schema metadata
    meta = schema.compile_parallel_metadata(tokenizer)
    field_items = meta["field_items"]
    suffix_lengths = meta["suffix_lengths"]
    candidate_token_sequences = meta["candidate_token_sequences"]
    suffix_token_lists = meta["suffix_token_lists"]
    suffixes_batch = meta["suffixes_batch"]
    M = len(field_items)

    # 2. High-density semantic catalog prefill
    system_prompt, user_prompt, assistant_prefill = build_parallel_chat(context, schema)
    base_toks = tokenize_chat_prefill(
        tokenizer,
        system_prompt,
        user_prompt,
        assistant_prefill,
        return_tensors="pt",
    )
    base_toks = base_toks.to(device)

    t_pre0 = time.perf_counter()
    with torch.no_grad():
        base_out = model(base_toks, use_cache=True)
        base_cache = base_out.past_key_values
    t_prefill = (time.perf_counter() - t_pre0) * 1000

    # 3. Parallel Suffix Evaluation
    t_suf0 = time.perf_counter()
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

    suffix_arr = torch.tensor(suffixes_batch, dtype=torch.long, device=device)
    suffix_mask = (suffix_arr != pad_id).long()

    # Broadcast KV cache to batch size M
    with torch.no_grad():
        batched_cache = copy.deepcopy(base_cache)
        if hasattr(batched_cache, "batch_repeat_interleave"):
            batched_cache.batch_repeat_interleave(M)
        elif isinstance(batched_cache, tuple):
            batched_cache = tuple(
                tuple(t.repeat(M, 1, 1, 1) for t in layer)
                for layer in batched_cache
            )

        prefix_len = base_toks.shape[1]
        prefix_mask = torch.ones((M, prefix_len), dtype=torch.long, device=device)
        full_mask = torch.cat([prefix_mask, suffix_mask], dim=1)

        out = model(suffix_arr, past_key_values=batched_cache, attention_mask=full_mask)
        suffix_out = out.logits

    t_suffix_eval = (time.perf_counter() - t_suf0) * 1000

    # 4. Length-normalized full-sequence scoring
    parsed_json = {}
    field_telemetry = {}
    t_sequence_scoring0 = time.perf_counter()
    continuation_calls = 0
    full_sequence_fields = 0
    max_sequence_tokens = 0

    for i, (fname, fdef) in enumerate(field_items):
        decision_idx = suffix_lengths[i] - 1
        field_logits = suffix_out[i, decision_idx, :]
        field_continuation_calls = 0
        sequences = candidate_token_sequences[i]
        max_sequence_tokens = max(max_sequence_tokens, max(map(len, sequences)))
        token_log_probabilities = [[] for _ in sequences]

        root_log_probs = F.log_softmax(field_logits.float() / max(temperature, 1e-4), dim=-1)
        for choice_index, sequence in enumerate(sequences):
            token_log_probabilities[choice_index].append(float(root_log_probs[sequence[0]].item()))

        continuation_indices = [index for index, sequence in enumerate(sequences) if len(sequence) > 1]
        if len(sequences) > 1 and continuation_indices:
            continuation_calls += 1
            field_continuation_calls = 1
            full_sequence_fields += 1
            input_rows = [
                suffix_token_lists[i] + sequences[index][:-1]
                for index in continuation_indices
            ]
            input_lengths = [len(row) for row in input_rows]
            max_input_length = max(input_lengths)
            padded_rows = [
                row + [pad_id] * (max_input_length - len(row))
                for row in input_rows
            ]
            continuation_ids = torch.tensor(padded_rows, dtype=torch.long, device=device)
            continuation_mask = torch.zeros_like(continuation_ids)
            for row_index, row_length in enumerate(input_lengths):
                continuation_mask[row_index, :row_length] = 1

            continuation_cache = copy.deepcopy(base_cache)
            batch_size = len(continuation_indices)
            if hasattr(continuation_cache, "batch_repeat_interleave"):
                continuation_cache.batch_repeat_interleave(batch_size)
            elif isinstance(continuation_cache, tuple):
                continuation_cache = tuple(
                    tuple(t.repeat(batch_size, 1, 1, 1) for t in layer)
                    for layer in continuation_cache
                )
            prefix_mask = torch.ones(
                (batch_size, base_toks.shape[1]),
                dtype=torch.long,
                device=device,
            )
            full_continuation_mask = torch.cat([prefix_mask, continuation_mask], dim=1)
            with torch.no_grad():
                continuation_out = model(
                    continuation_ids,
                    past_key_values=continuation_cache,
                    attention_mask=full_continuation_mask,
                    use_cache=False,
                )

            suffix_length = len(suffix_token_lists[i])
            for row_index, choice_index in enumerate(continuation_indices):
                sequence = sequences[choice_index]
                for token_index in range(1, len(sequence)):
                    logits_index = suffix_length + token_index - 1
                    logits = continuation_out.logits[row_index, logits_index, :].float()
                    logits /= max(temperature, 1e-4)
                    token_id = sequence[token_index]
                    token_log_prob = logits[token_id] - torch.logsumexp(logits, dim=-1)
                    token_log_probabilities[choice_index].append(float(token_log_prob.item()))

        all_probs, sequence_scores = score_full_sequences(token_log_probabilities)
        w_idx = int(max(range(len(all_probs)), key=all_probs.__getitem__))
        w_prob = float(all_probs[w_idx])

        if fdef.field_type == "boolean":
            val = (w_idx == 0)
        else:
            val = fdef.choices[w_idx]

        parsed_json[fname] = {
            "value": val,
            "prob": round(w_prob, 4)
        }

        choices_list = ["true", "false"] if fdef.field_type == "boolean" else fdef.choices
        scored_choices = []
        for c, p in zip(choices_list, all_probs):
            scored_choices.append({"choice": c, "probability": round(p, 4)})
        scored_choices.sort(key=lambda x: x["probability"], reverse=True)
        probabilities = {str(c): p for c, p in zip(choices_list, all_probs)}
        scores = {str(c): score for c, score in zip(choices_list, sequence_scores)}

        field_telemetry[fname] = {
            "value": val,
            "type": fdef.field_type,
            "confidence": round(w_prob, 4),
            "cardinality": fdef.cardinality,
            "probabilities": probabilities,
            "sequence_scores": scores,
            "continuation_calls": field_continuation_calls,
            "top_choices": scored_choices[:5]
        }

    t_sequence_scoring = (time.perf_counter() - t_sequence_scoring0) * 1000

    total_elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "mode": "parallel_constrained_calibrated",
        "elapsed_ms": round(total_elapsed_ms, 2),
        "prefill_ms": round(t_prefill, 2),
        "suffix_eval_ms": round(t_suffix_eval, 2),
        "sequence_scoring_ms": round(t_sequence_scoring, 2),
        "total_tokens_generated": 0,
        "sequential_forward_passes": 1 + continuation_calls,
        "continuation_calls": continuation_calls,
        "full_sequence_fields": full_sequence_fields,
        "max_sequence_tokens": max_sequence_tokens,
        "scoring_method": "mean_token_logprob",
        "is_valid_json": True,
        "schema_match": True,
        "parsed_json": parsed_json,
        "field_telemetry": field_telemetry,
        "has_calibrated_probabilities": True,
        "num_fields": len(schema),
        "device": device
    }


def run_naive_generation_torch(
    context: str,
    schema: StructuredSchema,
    temperature: float = 0.2,
    max_new_tokens: int = 512
) -> Dict[str, Any]:
    """
    Standard autoregressive baseline using PyTorch.
    """
    model, tokenizer, device = get_torch_engine()
    t0 = time.perf_counter()

    system_prompt, user_prompt, assistant_prefill = build_naive_chat(context, schema)
    input_ids = tokenize_chat_prefill(
        tokenizer,
        system_prompt,
        user_prompt,
        assistant_prefill,
        return_tensors="pt",
    ).to(device)
    prompt_tokens = input_ids.shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0.0),
            temperature=max(temperature, 1e-4),
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )

    elapsed_ms = (time.perf_counter() - t0) * 1000
    gen_tokens = output_ids.shape[1] - prompt_tokens
    tok_per_sec = (gen_tokens / (elapsed_ms / 1000.0)) if elapsed_ms > 0 else 0.0

    raw_text = assistant_prefill + tokenizer.decode(
        output_ids[0][prompt_tokens:],
        skip_special_tokens=True,
    )

    # Parse JSON
    parsed_json = None
    is_valid = False
    try:
        first_brace = raw_text.find("{")
        last_brace = raw_text.rfind("}")
        if first_brace != -1 and last_brace != -1:
            cleaned = raw_text[first_brace:last_brace + 1]
            parsed_json = json.loads(cleaned)
            is_valid = True
    except Exception:
        pass

    schema_match = False
    if is_valid and isinstance(parsed_json, dict):
        expected_keys = set(schema.get_field_names())
        schema_match = (set(parsed_json.keys()) == expected_keys)

    return {
        "mode": "autoregressive_naive",
        "elapsed_ms": round(elapsed_ms, 2),
        "total_tokens": gen_tokens,
        "tokens_per_second": round(tok_per_sec, 1),
        "sequential_forward_passes": gen_tokens,
        "is_valid_json": is_valid,
        "schema_match": schema_match,
        "raw_text": raw_text,
        "parsed_json": parsed_json,
        "device": device
    }


def stream_naive_generation_torch(
    context: str,
    schema: StructuredSchema,
    temperature: float = 0.2,
    max_new_tokens: int = 512
) -> Generator[Dict[str, Any], None, None]:
    """
    Generator streaming individual tokens for side-by-side comparison visualizer.
    """
    model, tokenizer, device = get_torch_engine()
    t0 = time.perf_counter()

    system_prompt, user_prompt, assistant_prefill = build_naive_chat(context, schema)
    input_ids = tokenize_chat_prefill(
        tokenizer,
        system_prompt,
        user_prompt,
        assistant_prefill,
        return_tensors="pt",
    ).to(device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    gen_kwargs = {
        "input_ids": input_ids,
        "max_new_tokens": max_new_tokens,
        "do_sample": (temperature > 0.0),
        "temperature": max(temperature, 1e-4),
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "streamer": streamer
    }

    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    full_text = assistant_prefill
    tok_count = 0

    for token_str in streamer:
        tok_count += 1
        full_text += token_str
        yield {
            "type": "token",
            "token": token_str,
            "token_count": tok_count
        }

    thread.join()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    parsed_json = None
    is_valid = False
    try:
        first_brace = full_text.find("{")
        last_brace = full_text.rfind("}")
        if first_brace != -1 and last_brace != -1:
            cleaned = full_text[first_brace:last_brace + 1]
            parsed_json = json.loads(cleaned)
            is_valid = True
    except Exception:
        pass

    schema_match = False
    if is_valid and isinstance(parsed_json, dict):
        expected_keys = set(schema.get_field_names())
        schema_match = (set(parsed_json.keys()) == expected_keys)

    result = {
        "mode": "autoregressive_naive",
        "elapsed_ms": round(elapsed_ms, 2),
        "total_tokens": tok_count,
        "tokens_per_second": round((tok_count / (elapsed_ms / 1000.0)) if elapsed_ms > 0 else 0.0, 1),
        "sequential_forward_passes": tok_count,
        "is_valid_json": is_valid,
        "schema_match": schema_match,
        "raw_text": full_text,
        "parsed_json": parsed_json,
        "device": device
    }

    yield {
        "type": "done",
        "result": result
    }
