"""Run a Wikipedia search through the local jev-browser HTTP service."""

import argparse
import json
import sys
from html.parser import HTMLParser
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_GOAL = (
    "Search Wikipedia for Gödel's incompleteness theorems and open the matching article. "
    "Stop when the article page is visibly open; then choose DONE."
)


class TokenParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.token = None

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "meta" and values.get("name") == "demo-token":
            self.token = values.get("content")


class JevBrowserClient:
    def __init__(self, base_url, timeout):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.token = self._load_token()

    def _load_token(self):
        with urlopen(f"{self.base_url}/", timeout=self.timeout) as response:
            parser = TokenParser()
            parser.feed(response.read().decode())
        if not parser.token:
            raise RuntimeError("jev-browser did not return a demo token")
        return parser.token

    def post(self, endpoint, body):
        request = Request(
            f"{self.base_url}/api/{endpoint}",
            data=json.dumps(body).encode(),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "X-Demo-Token": self.token,
            },
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except HTTPError as error:
            try:
                detail = json.loads(error.read()).get("error", error.reason)
            except (json.JSONDecodeError, UnicodeDecodeError):
                detail = error.reason
            raise RuntimeError(f"jev-browser returned HTTP {error.code}: {detail}") from error


def page_summary(state):
    page = state.get("page") or {}
    return {
        "status": state.get("status"),
        "url": page.get("url"),
        "title": page.get("title"),
        "actions": len(state.get("history", [])),
        "elapsed_ms": state.get("elapsed_ms", 0),
    }


def decision_summary(decision):
    return {
        key: decision.get(key)
        for key in (
            "model",
            "operation",
            "target",
            "choice",
            "confidence",
            "target_confidence",
            "latency_ms",
            "operation_probabilities",
            "target_probabilities",
        )
    }


def print_tick_result(state, previous_actions, tick_number, decision):
    history = state.get("history", [])
    if len(history) > previous_actions:
        for action in history[previous_actions:]:
            print(f"[action {action['step']:02d}] {state['status']}: {action['action']}")
        return
    if state["status"] in {"done", "blocked"}:
        choice = decision.get("choice", state["status"].upper()) if decision else state["status"].upper()
        print(f"[terminal] {state['status']}: {choice} ({len(history)} browser actions)")
        return
    choice = decision.get("choice", "decision") if decision else "decision"
    print(f"[tick {tick_number:02d}] {state['status']}: {choice} not executed; page changed and was re-observed")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-url", default="http://127.0.0.1:8766", help="jev-browser service URL")
    parser.add_argument(
        "--target-url",
        default="https://en.wikipedia.org/wiki/Main_Page",
        help="HTTPS page where the browser agent starts",
    )
    parser.add_argument("--goal", default=DEFAULT_GOAL)
    parser.add_argument("--timeout", type=float, default=150, help="Timeout for each HTTP request")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        client = JevBrowserClient(args.service_url, args.timeout)
        state = client.post("reset", {"url": args.target_url, "goal": args.goal})
        print("Browser ready:", json.dumps(page_summary(state), ensure_ascii=False))

        logged_decisions = 0
        for tick_number in range(1, state["max_steps"] * 2 + 1):
            if state["status"] in {"done", "blocked"}:
                break
            previous_actions = len(state.get("history", []))
            state = client.post("tick", {})
            decisions = state.get("decisions", [])
            new_decisions = decisions[logged_decisions:]
            for index, decision in enumerate(new_decisions, start=logged_decisions + 1):
                print(
                    f"[model {index:02d}] "
                    + json.dumps(decision_summary(decision), ensure_ascii=False, sort_keys=True)
                )
            logged_decisions = len(decisions)
            print_tick_result(state, previous_actions, tick_number, new_decisions[-1] if new_decisions else None)

        print("Final:", json.dumps(page_summary(state), ensure_ascii=False))
        return 0 if state["status"] == "done" else 1
    except (HTTPError, URLError, RuntimeError, TimeoutError, KeyError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
