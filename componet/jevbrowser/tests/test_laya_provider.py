import pytest

from jev_ultrafast import demo, model


def test_laya_provider_posts_system_one_request(monkeypatch):
    captured = {}

    def fake_post(url, key, body, timeout=None):
        captured.update(url=url, key=key, body=body, timeout=timeout)
        return {
            "model": "rl-agent",
            "answers": {
                "operation": {
                    "type": "choice",
                    "choice": "WAIT",
                    "probabilities": {"WAIT": 0.6, "DONE": 0.2, "BLOCKED": 0.2},
                    "confidence": 0.4,
                }
            },
            "usage": {"input_tokens": 42, "output_tokens": 0},
        }

    monkeypatch.setattr(model, "post_json", fake_post)
    monkeypatch.setenv("JEV_MODEL_PROVIDER", "laya")
    monkeypatch.setenv("JEV_MODEL_BASE_URL", "http://laya:8000")
    monkeypatch.setenv("JEV_MODEL_API_KEY", "secret")
    monkeypatch.setenv("JEV_MODEL_TIMEOUT", "15")

    result = model.choose(
        {
            "url": "https://example.com",
            "title": "Example",
            "text": "Example page",
            "actions": [{"kind": "wait", "id": "wait", "label": "Wait"}],
        },
        "Wait once",
        [],
    )

    assert result["operation"] == "WAIT"
    assert result["choice"] == "wait"
    assert captured["url"] == "http://laya:8000/v1/systemone"
    assert captured["key"] == "secret"
    assert captured["timeout"] == 15
    assert set(captured["body"]) == {"state", "questions"}
    assert captured["body"]["questions"]["operation"]["criteria"]["WAIT"]


@pytest.mark.parametrize("value", ["0", "-1", "not-a-number"])
def test_laya_provider_rejects_invalid_timeout(monkeypatch, value):
    monkeypatch.setenv("JEV_MODEL_PROVIDER", "laya")
    monkeypatch.setenv("JEV_MODEL_BASE_URL", "http://laya:8000")
    monkeypatch.setenv("JEV_MODEL_TIMEOUT", value)

    with pytest.raises(ValueError, match="JEV_MODEL_TIMEOUT"):
        model.choose(
            {
                "url": "https://example.com",
                "title": "Example",
                "text": "Example page",
                "actions": [{"kind": "wait", "id": "wait", "label": "Wait"}],
            },
            "Wait once",
            [],
        )
def test_wikipedia_scenario_uses_public_main_page():
    assert demo.scenario_url("wikipedia") == "https://en.wikipedia.org/wiki/Main_Page"


def test_unknown_demo_scenario_is_rejected():
    with pytest.raises(ValueError, match="Unknown demo scenario"):
        demo.scenario_url("unknown")


def test_arbitrary_url_requires_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("JEV_ALLOW_ARBITRARY_URLS", raising=False)

    with pytest.raises(ValueError, match="JEV_ALLOW_ARBITRARY_URLS=1"):
        demo.start_url({"url": "https://en.wikipedia.org/wiki/Main_Page"})


def test_arbitrary_url_accepts_public_https_when_enabled(monkeypatch):
    monkeypatch.setenv("JEV_ALLOW_ARBITRARY_URLS", "1")
    monkeypatch.setattr(
        demo.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [(demo.socket.AF_INET, demo.socket.SOCK_STREAM, 6, "", ("208.80.154.224", 443))],
    )

    url = "https://en.wikipedia.org/wiki/Main_Page"
    assert demo.start_url({"url": url}) == url


@pytest.mark.parametrize(
    "url, message",
    [
        ("http://en.wikipedia.org/wiki/Main_Page", "absolute HTTPS"),
        ("https://user:password@example.com/", "credentials"),
        ("https://127.0.0.1/", "public IP"),
        ("https://169.254.169.254/", "public IP"),
    ],
)
def test_arbitrary_url_rejects_unsafe_targets(monkeypatch, url, message):
    monkeypatch.setenv("JEV_ALLOW_ARBITRARY_URLS", "1")

    with pytest.raises(ValueError, match=message):
        demo.start_url({"url": url})
