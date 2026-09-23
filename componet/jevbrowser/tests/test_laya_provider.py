import pytest

from jev_ultrafast import model


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
