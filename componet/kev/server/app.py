"""FastAPI wrapper around Kev's System One inference pipeline."""

import os
import threading
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import Any

import torch
from fastapi import FastAPI, HTTPException, Request

from .api import SystemOneRequest, output_tokens, to_answers, to_record, with_date_facts
from .checkpoint import Checkpoint, LoadOptions
from .model import SERVE_MAX_BRANCH, SERVE_MAX_STATE


class KevAgent:
    """A loaded Kev checkpoint with serialized access to model inference."""

    def __init__(self, checkpoint: Checkpoint, tokenizer: Any, model: Any):
        self.checkpoint = checkpoint
        self.tokenizer = tokenizer
        self.model = model
        self.inference_lock = threading.Lock()
        self.capture_lock = threading.Lock()

    def system_one(self, request: SystemOneRequest) -> dict[str, Any]:
        if os.environ.get("KEV_DATE_FACTS", "0") == "1":
            request = request.model_copy(update={"state": with_date_facts(request.state)})

        record, metadata = to_record(request)
        try:
            encoded = self.model.encode(
                self.tokenizer,
                record,
                max_state=SERVE_MAX_STATE,
                max_branch=SERVE_MAX_BRANCH,
                strict=True,
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        with self.inference_lock:
            if getattr(self.model, "graphs", None) is not None:
                probabilities = self.model.probs_and_prefix(encoded)[0]
            else:
                probabilities = self.model.probs(encoded)
        self.capture_pending()

        distributions = [probability.tolist() for probability in probabilities]
        answers = to_answers(distributions, metadata)
        return {
            "model": request.model,
            "answers": answers,
            "usage": {
                "input_tokens": len(encoded["ids"]),
                "output_tokens": output_tokens(self.tokenizer, answers),
            },
        }

    def capture_pending(self) -> None:
        """Capture newly seen CUDA graph buckets after the response path releases the model lock."""
        graphs = getattr(self.model, "graphs", None)
        if graphs is None or not graphs.pending or not self.capture_lock.acquire(blocking=False):
            return

        def capture() -> None:
            try:
                while graphs.pending:
                    with self.inference_lock:
                        graphs.capture_pending(limit=1)
            finally:
                self.capture_lock.release()

        threading.Thread(target=capture, daemon=True).start()


def default_device() -> str:
    """Select the best locally available torch device."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_agent() -> KevAgent:
    """Load the configured checkpoint once when the server starts."""
    model_id = os.environ.get("KEV_MODEL_ID", "jaredpalmer/kev-4b")
    device = os.environ.get("KEV_DEVICE") or default_device()
    options = LoadOptions.from_env()

    # Kev checkpoints are evaluated in fp32 by default, but bf16 is the practical
    # serving default on accelerators unless the caller explicitly selected a dtype.
    if device != "cpu" and options.dtype is None:
        options = replace(options, dtype=torch.bfloat16)
    # The CUDA graph implementation lives in server/cuda_graphs.py. The
    # quantized path is supported experimentally and can be disabled with 0.
    if str(device).startswith("cuda") and options.cuda_graphs is None:
        options = replace(options, cuda_graphs=True)

    checkpoint = Checkpoint(model_id)
    tokenizer, model = checkpoint.load(device, options)
    return KevAgent(checkpoint, tokenizer, model)


def warmup(agent: KevAgent) -> None:
    """Run a minimal decision so lazy model kernels are ready before startup ends."""
    agent.system_one(
        SystemOneRequest(
            state={"service": "kev", "status": "warming up"},
            questions={
                "ready": {
                    "type": "noul",
                    "instructions": "Is this a Kev service warm-up request?",
                    "criteria": {
                        "true": "The state explicitly says the service is warming up.",
                        "false": "The state is unrelated to service warm-up.",
                    },
                }
            },
        )
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    agent = load_agent()
    warmup(agent)
    app.state.agent = agent
    yield


app = FastAPI(
    title="Kev System One API",
    description="HTTP API for typed decisions using a Kev checkpoint.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", tags=["Health"])
def health(request: Request):
    return {"status": "ok", "model_loaded": hasattr(request.app.state, "agent")}


@app.post("/v1/systemone")
@app.post("/api/system-one", include_in_schema=False)
def system_one(body: SystemOneRequest, request: Request):
    return request.app.state.agent.system_one(body)
