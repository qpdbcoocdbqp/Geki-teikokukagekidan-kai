"""FastAPI wrapper around Laya's ``system_one`` inference call."""

import json
import os
import threading
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field


class SystemOneRequest(BaseModel):
    """Accept native Laya and legacy RLCD request bodies."""

    model_config = ConfigDict(populate_by_name=True)

    state: Any = None
    questions: dict[str, dict[str, Any]] | None = None
    context: Any = None
    schema_def: dict[str, dict[str, Any]] | None = Field(default=None, alias="schema")
    temperature: float | None = None

    def to_laya(self):
        native = "state" in self.model_fields_set or "questions" in self.model_fields_set
        legacy = "context" in self.model_fields_set or "schema_def" in self.model_fields_set
        if native and legacy:
            raise ValueError("Use either state/questions or context/schema, not both")
        if native:
            if "state" not in self.model_fields_set or not self.questions:
                raise ValueError("Laya requests require state and non-empty questions")
            return self.state, self.questions
        if not legacy or "context" not in self.model_fields_set or not self.schema_def:
            raise ValueError("Request requires state/questions or context/schema")

        state = self.context
        if isinstance(state, str):
            try:
                state = json.loads(state)
            except json.JSONDecodeError:
                pass

        questions = {}
        for name, definition in self.schema_def.items():
            field_type = str(definition.get("type", "")).lower()
            if field_type == "boolean":
                choices = ["true", "false"]
            elif field_type in {"enum", "choice", "selection"}:
                choices = definition.get("choices")
                if not isinstance(choices, list) or not choices:
                    raise ValueError(f"schema field {name!r} requires non-empty choices")
            else:
                raise ValueError(f"schema field {name!r} has unsupported type {field_type!r}")
            labels = [str(choice) for choice in choices]
            questions[name] = {
                "type": "choice",
                "instructions": definition.get("description") or f"Choose {name}.",
                "criteria": {label: label for label in labels},
            }
        return state, questions


def load_agent():
    """Load the configured checkpoint once when the server starts."""
    import laya

    model_id = os.environ.get("LAYA_MODEL_ID", "convaiinnovations/laya")
    device = os.environ.get("LAYA_DEVICE", "cuda")
    subfolder = os.environ.get("LAYA_MODEL_SUBFOLDER")
    options = {"device": device}
    if subfolder:
        options["subfolder"] = subfolder
    return laya.load(model_id, **options)


def warmup(agent):
    """Compile lazy model kernels before the service reports healthy."""
    agent.system_one(
        {"service": "laya", "status": "warming up"},
        {
            "ready": {
                "type": "noul",
                "instructions": "Is this a Laya service warm-up request?",
                "criteria": {
                    "true": "The state explicitly says the service is warming up.",
                    "false": "The state is unrelated to service warm-up.",
                },
            }
        },
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    agent = load_agent()
    warmup(agent)
    app.state.agent = agent
    app.state.inference_lock = threading.Lock()
    yield


app = FastAPI(
    title="Laya System One API",
    description="HTTP API for batched typed decisions using Laya system_one.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", tags=["Health"])
def health(request: Request):
    return {"status": "ok", "model_loaded": hasattr(request.app.state, "agent")}


@app.post("/v1/systemone")
@app.post("/api/system-one", include_in_schema=False)
def system_one(body: SystemOneRequest, request: Request):
    try:
        state, questions = body.to_laya()
        with request.app.state.inference_lock:
            return request.app.state.agent.system_one(state, questions)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
