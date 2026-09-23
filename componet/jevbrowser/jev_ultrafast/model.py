"""A configured policy model makes choices; an optional model writes field values."""

import json
import math
import os
import time

import httpx

from .questions import NEXT_ACTION, TARGET, TEXT_VALUE

CLIENT = httpx.Client(http2=True, timeout=25)


def post_json(url, key, body, timeout=None):
    for attempt in range(3):
        try:
            options = {} if timeout is None else {"timeout": timeout}
            headers = {"Authorization": f"Bearer {key}"} if key else {}
            response = CLIENT.post(url, json=body, headers=headers, **options)
        except httpx.TimeoutException:
            raise RuntimeError("Model request timed out; no action executed.") from None
        except httpx.HTTPError:
            raise RuntimeError("Model connection failed; no action executed.") from None
        if response.status_code in {429, 529, 503} and attempt < 2:
            time.sleep(0.5 * 2**attempt)
            continue
        if response.is_error:
            raise RuntimeError(f"Model provider returned HTTP {response.status_code}; no action executed.")
        return response.json()
    raise RuntimeError("Model unavailable")


def validate_choice(answer, ids, source="TypeSafe"):
    try:
        probabilities = answer["probabilities"]
        numbers = [*probabilities.values(), answer["confidence"]]
        valid = (
            answer["choice"] in ids
            and set(probabilities) == set(ids)
            and all(type(n) in (int, float) and math.isfinite(n) and 0 <= n <= 1 for n in numbers)
            and abs(sum(probabilities.values()) - 1) < 0.02
            and probabilities[answer["choice"]] >= max(probabilities.values()) - 1e-6
        )
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError(f"Invalid {source} response; no action executed.")
    return answer


def action_space(actions):
    """One index per observed element; each operation has its own valid target choices."""
    elements, indices, targets, controls = [], {}, {}, {}
    operations = {"click": "CLICK", "fill": "TYPE_TEXT", "select": "SELECT"}
    for action in actions:
        kind = action["kind"]
        if kind not in operations:
            controls[action["id"].upper()] = action
            continue
        node = action["node"]
        if node not in indices:
            index = str(len(elements) + 1)
            indices[node] = index
            element = {k: action[k] for k in ("role", "value", "checked", "selected", "expanded") if k in action}
            element.update(index=index, label=action["label"].split(" → ")[0], operations=[])
            if kind == "select":
                element["value"] = action.get("current_value", "")
                element["options"] = []
            elements.append(element)
        index = indices[node]
        operation = operations[kind]
        group = targets.setdefault(operation, {})
        element = elements[int(index) - 1]
        if operation not in element["operations"]:
            element["operations"].append(operation)
        target = index
        if kind == "select":
            target = f"{index}:{len(element['options']) + 1}"
            element["options"].append({"index": target, "label": action["label"], "value": action["value"]})
        group[target] = action
    return elements, targets, controls


def openai_choice(state, goal, history, operations, targets, controls):
    base = os.environ.get("JEV_MODEL_BASE_URL", os.environ.get("TEXT_MODEL_BASE_URL", "")).rstrip("/")
    model = os.environ.get("JEV_MODEL", os.environ.get("TEXT_MODEL"))
    key = os.environ.get("JEV_MODEL_API_KEY", os.environ.get("TEXT_MODEL_API_KEY", "local"))
    if not base or not model:
        raise ValueError("OpenAI policy needs JEV_MODEL_BASE_URL and JEV_MODEL; no action executed.")
    try:
        timeout = float(os.environ.get("JEV_MODEL_TIMEOUT", "120"))
        if timeout <= 0:
            raise ValueError()
    except ValueError:
        raise ValueError("JEV_MODEL_TIMEOUT must be a positive number; no action executed.") from None
    request = {
        "goal": goal,
        "page": {k: state[k] for k in ("url", "title", "text")},
        "recent_actions": [
            {k: item.get(k) for k in ("action", "kind", "text", "page_changed")} for item in history[-10:]
        ],
        "allowed_operations": operations,
        "allowed_targets": {
            operation: {
                index: {
                    "element": f"[{index}] {action['label']}",
                    "current_value": action.get("current_value", action.get("value", "")),
                    **{
                        key: action[key]
                        for key in ("role", "checked", "selected", "expanded")
                        if key in action
                    },
                }
                for index, action in candidates.items()
            }
            for operation, candidates in targets.items()
        },
    }
    body = {
        "model": model,
        "temperature": 0,
        "max_tokens": 256,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "Choose exactly one next browser action from the supplied allowlists. Return only a JSON object "
                    'with exactly {"operation":"...","target":"..."}. Use an exact allowed operation. When that '
                    "operation has allowed targets, use one exact target key; otherwise target must be null. Never "
                    "invent selectors, coordinates, text, code, operations, or targets. Choose DONE only when every "
                    "goal requirement is visibly satisfied; choose BLOCKED only when no allowed action can progress."
                ),
            },
            {"role": "user", "content": json.dumps(request)},
        ],
    }
    result = post_json(base + "/chat/completions", key, body, timeout=timeout)
    content = ""
    try:
        content = result["choices"][0]["message"]["content"]
        answer = json.loads(content)
        operation = answer["operation"]
        target = answer["target"]
        valid = set(answer) == {"operation", "target"} and operation in operations
        if operation in targets:
            valid = valid and target in targets[operation]
        else:
            valid = valid and target is None
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        valid = False
    if not valid:
        detail = content.replace("\n", " ")[:300]
        raise ValueError(f"Invalid OpenAI policy response {detail!r}; no action executed.")

    operation_probability = 1 / len(operations)
    operation_probabilities = {name: operation_probability for name in operations}
    if operation in targets:
        target_probability = 1 / len(targets[operation])
        target_probabilities = {index: target_probability for index in targets[operation]}
        choice = targets[operation][target]["id"]
        probabilities = {action["id"]: target_probability for action in targets[operation].values()}
    else:
        target_probabilities = {}
        choice = controls[operation]["id"] if operation in controls else operation
        probabilities = {choice: operation_probability}
    return {
        "choice": choice,
        "operation": operation,
        "target": target,
        "confidence": 0.0,
        "probabilities": probabilities,
        "operation_probabilities": operation_probabilities,
        "target_probabilities": target_probabilities,
        "target_confidence": 0.0 if target is not None else None,
        "raw_answers": answer,
        "model": result.get("model", model),
        "usage": result.get("usage", {}),
        "request": body,
    }


def rlcd_choice(state, goal, history, operations, targets, controls):
    base = os.environ.get("JEV_MODEL_BASE_URL", "").rstrip("/")
    key = os.environ.get("JEV_MODEL_API_KEY", "local")
    if not base:
        raise ValueError("RLCD policy needs JEV_MODEL_BASE_URL; no action executed.")
    try:
        timeout = float(os.environ.get("JEV_MODEL_TIMEOUT", "120"))
        if timeout <= 0:
            raise ValueError()
    except ValueError:
        raise ValueError("JEV_MODEL_TIMEOUT must be a positive number; no action executed.") from None

    request = {
        "goal": goal,
        "page": {k: state[k] for k in ("url", "title", "text")},
        "recent_actions": [
            {k: item.get(k) for k in ("action", "kind", "text", "page_changed")} for item in history[-10:]
        ],
        "allowed_operations": operations,
        "allowed_targets": {
            operation: {
                index: {
                    "element": f"[{index}] {action['label']}",
                    "current_value": action.get("current_value", action.get("value", "")),
                    **{
                        name: action[name]
                        for name in ("role", "checked", "selected", "expanded")
                        if name in action
                    },
                }
                for index, action in candidates.items()
            }
            for operation, candidates in targets.items()
        },
    }
    schema = {
        "operation": {
            "type": "enum",
            "choices": list(operations),
            "description": "The single best next browser operation from allowed_operations in the context.",
        }
    }
    for operation, candidates in targets.items():
        schema[operation.lower() + "_target"] = {
            "type": "enum",
            "choices": list(candidates),
            "description": (
                f"The best observed target for {operation}; use the exact key from "
                f"allowed_targets.{operation} in the context."
            ),
        }
    body = {"context": json.dumps(request), "schema": schema, "temperature": 1.0}
    endpoint = base if base.endswith("/api/run-rlcd") else base + "/api/run-rlcd"
    result = post_json(endpoint, key, body, timeout=timeout)

    def answer(field, ids):
        try:
            parsed = result["parsed_json"][field]
            telemetry = result["field_telemetry"][field]
            normalized = {
                "choice": parsed["value"],
                "confidence": telemetry["confidence"],
                "probabilities": telemetry["probabilities"],
            }
        except (KeyError, TypeError):
            normalized = {}
        return validate_choice(normalized, ids, source="RLCD")

    operation_answer = answer("operation", operations)
    operation = operation_answer["choice"]
    target = None
    target_answer = None
    probabilities = {}
    if operation in targets:
        target_answer = answer(operation.lower() + "_target", targets[operation])
        target = target_answer["choice"]
        choice = targets[operation][target]["id"]
        probabilities = {
            action["id"]: target_answer["probabilities"][index]
            for index, action in targets[operation].items()
        }
    else:
        choice = controls[operation]["id"] if operation in controls else operation
        probabilities[choice] = operation_answer["probabilities"][operation]
    return {
        "choice": choice,
        "operation": operation,
        "target": target,
        "confidence": operation_answer["confidence"],
        "probabilities": probabilities,
        "operation_probabilities": operation_answer["probabilities"],
        "target_probabilities": target_answer["probabilities"] if target_answer else {},
        "target_confidence": target_answer["confidence"] if target_answer else None,
        "raw_answers": result.get("parsed_json", {}),
        "model": result.get("model", os.environ.get("JEV_MODEL", "rlcd")),
        "usage": result.get("usage", {}),
        "request": body,
    }


def choose(state, goal, history):
    started = time.perf_counter()
    elements, targets, controls = action_space(state["actions"])
    labels = {
        "CLICK": "Click an element, button, menu option, autocomplete suggestion, or calendar day.",
        "TYPE_TEXT": "Enter or replace text in an editable field. A small LLM will supply the value from the goal.",
        "SELECT": "Select an observed dropdown value.",
    }
    operations = {key: labels[key] for key in targets}
    operations.update({key: value["label"] for key, value in controls.items()})
    operations.update(DONE="Every requirement is visibly satisfied.", BLOCKED="No supported operation can progress.")
    provider = os.environ.get("JEV_MODEL_PROVIDER", "typesafe").lower()
    if provider == "openai":
        decision = openai_choice(state, goal, history, operations, targets, controls)
        decision["latency_ms"] = round((time.perf_counter() - started) * 1000)
        return decision
    if provider == "rlcd":
        decision = rlcd_choice(state, goal, history, operations, targets, controls)
        decision["latency_ms"] = round((time.perf_counter() - started) * 1000)
        return decision
    if provider not in {"typesafe", "laya"}:
        raise ValueError(
            f"Unknown JEV_MODEL_PROVIDER {provider!r}; expected 'typesafe', 'openai', 'rlcd', or 'laya'."
        )
    questions = {
        "operation": {"type": "choice", "criteria": operations, "instructions": {"goal": goal, "rules": NEXT_ACTION}}
    }
    for operation, candidates in targets.items():
        questions[operation.lower() + "_target"] = {
            "type": "choice",
            "criteria": {
                index: {
                    "element": f"[{index}] {a['label']}",
                    "current_value": a.get("current_value", a.get("value", "")),
                    **{k: a[k] for k in ("role", "checked", "selected", "expanded") if k in a},
                }
                for index, a in candidates.items()
            },
            "instructions": {"goal": goal, "operation": operation, "rules": [NEXT_ACTION, TARGET]},
        }
    body = {
        "model": os.environ.get("TYPESAFE_MODEL", "jev-latest"),
        "state": {
            "page": {k: state[k] for k in ("url", "title", "text")},
            "elements": elements,
            "recent_actions": [
                {k: h.get(k) for k in ("action", "kind", "text", "page_changed")} for h in history[-10:]
            ],
        },
        "questions": questions,
    }
    if provider == "laya":
        base = os.environ.get("JEV_MODEL_BASE_URL", "").rstrip("/")
        if not base:
            raise ValueError("Laya policy needs JEV_MODEL_BASE_URL; no action executed.")
        endpoint = base if base.endswith(("/v1/systemone", "/api/system-one")) else base + "/v1/systemone"
        key = os.environ.get("JEV_MODEL_API_KEY", "")
        try:
            timeout = float(os.environ.get("JEV_MODEL_TIMEOUT", "120"))
            if timeout <= 0:
                raise ValueError()
        except ValueError:
            raise ValueError("JEV_MODEL_TIMEOUT must be a positive number; no action executed.") from None
        result = post_json(endpoint, key, {"state": body["state"], "questions": questions}, timeout=timeout)
        source = "Laya"
    else:
        result = post_json("https://api.typesafe.ai/v1/systemone", os.environ["TYPESAFE_API_KEY"], body)
        source = "TypeSafe"
    operation_answer = validate_choice(result["answers"].get("operation", {}), operations, source=source)
    operation = operation_answer["choice"]
    target = None
    target_answer = None
    probabilities = {}
    if operation in targets:
        # Unused target heads cannot cause an action. Validate the head selected by the operation.
        target_answer = validate_choice(
            result["answers"].get(operation.lower() + "_target", {}), targets[operation], source=source
        )
        target = target_answer["choice"]
        choice = targets[operation][target]["id"]
        probabilities = {a["id"]: target_answer["probabilities"][index] for index, a in targets[operation].items()}
    else:
        choice = controls[operation]["id"] if operation in controls else operation
        probabilities[choice] = operation_answer["probabilities"][operation]
    return {
        "choice": choice,
        "operation": operation,
        "target": target,
        "confidence": operation_answer["confidence"],
        "probabilities": probabilities,
        "operation_probabilities": operation_answer["probabilities"],
        "target_probabilities": target_answer["probabilities"] if target_answer else {},
        "target_confidence": target_answer["confidence"] if target_answer else None,
        "raw_answers": result["answers"],
        "model": result["model"],
        "usage": result.get("usage", {}),
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "request": body,
    }


def field_context(goal, action, page, history):
    return {
        "goal": goal,
        "field": {k: action.get(k) for k in ("label", "role", "value")},
        "page": {"title": page["title"], "text": page["text"][:6000]},
        "recent_actions": [{k: h.get(k) for k in ("action", "text")} for h in history[-6:]],
    }


def field_text(context):
    key = os.environ.get("TEXT_MODEL_API_KEY")
    if not key:
        raise ValueError("TYPE_TEXT needs TEXT_MODEL_API_KEY; no text is hardcoded or guessed by the executor.")
    base = os.environ.get("TEXT_MODEL_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
    model = os.environ.get("TEXT_MODEL", "deepseek-chat")
    reasoning = {"thinking": {"type": "disabled"}} if "api.deepseek.com/" in base else {"reasoning": {"effort": "low"}}
    if os.environ.get("TEXT_MODEL_REASONING") == "none":
        reasoning = {"reasoning": {"enabled": False}}
    started = time.perf_counter()
    result = post_json(
        base + "/chat/completions",
        key,
        {
            "model": model,
            "max_tokens": 1024,
            "response_format": {"type": "json_object"},
            **reasoning,
            "messages": [
                {"role": "system", "content": TEXT_VALUE},
                {
                    "role": "user",
                    "content": json.dumps(context),
                },
            ],
        },
    )
    try:
        output = json.loads(result["choices"][0]["message"]["content"])
        value = output["text"]
        if set(output) != {"text"} or not isinstance(value, str) or not value.strip() or len(value) > 2000:
            raise ValueError()
    except (ValueError, KeyError, TypeError):
        raise ValueError("Text helper returned no valid field value; nothing typed.") from None
    return value, {
        "model": model,
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "usage": result.get("usage", {}),
    }
