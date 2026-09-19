"""
Prompt construction utilities for Autoregressive JSON Generation
vs. Parallel Constrained Decision Batches.
"""

from typing import List, Tuple
from core.schema import StructuredSchema, FieldDefinition


def build_naive_chat(context: str, schema: StructuredSchema):
    """Build model-neutral chat content and an assistant JSON prefill."""
    schema_prompt = schema.to_json_schema_prompt_str()
    system = (
        "You are a precise data extraction system. Output only a valid, indented JSON object "
        "matching the schema. Do not include markdown.\n\n"
        f"JSON Schema:\n{schema_prompt}"
    )
    user = f"Analyze the following context and produce the required JSON object:\n\n{context}"
    return system, user, '{\n  "'


def build_naive_json_prompt(context: str, schema: StructuredSchema) -> str:
    """Build a readable legacy prompt without model-specific control tokens."""
    system, user, assistant_prefill = build_naive_chat(context, schema)
    return f"{system}\n\n{user}\n\n{assistant_prefill}"


def build_parallel_chat(context: str, schema: StructuredSchema):
    """Build model-neutral content for the shared parallel-decision prefill."""
    schema_str = schema.to_parallel_schema_str()
    system = f"Classify JSON attributes using the supplied evidence:\n{schema_str}"
    return system, context, "{\n"


def build_parallel_field_prompts(context: str, schema: StructuredSchema) -> List[Tuple[str, FieldDefinition, str]]:
    """
    Builds discrete single-decision prompts for each field in the schema.
    Returns a list of (field_name, field_def, prompt_text).
    """
    prompts = []
    for field_name, field_def in schema.fields.items():
        if field_def.field_type == "boolean":
            options_text = "true, false"
        else:
            if len(field_def.choices) <= 20:
                options_text = ", ".join(field_def.choices)
            else:
                sample = ", ".join(field_def.choices[:8])
                options_text = f"{sample}, ... [{len(field_def.choices)} total options]"

        prompt = (
            "You are a calibrated decision engine. Select the single most accurate option based on evidence.\n\n"
            f"{context}\n\n"
            f"Field: {field_name}\n"
            f"Description: {field_def.description}\n"
            f"Allowed choices: {options_text}\n"
            "Exact choice:"
        )
        prompts.append((field_name, field_def, prompt))
        
    return prompts


# Backward compatibility alias
build_rlcd_field_prompts = build_parallel_field_prompts
