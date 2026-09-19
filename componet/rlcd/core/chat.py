"""Model-agnostic chat-template rendering with an assistant prefill."""


def tokenize_chat_prefill(tokenizer, system, user, assistant_prefill, return_tensors="pt"):
    """Render a chat using the tokenizer's own template and continue the prefill.

    Some instruction models do not accept a dedicated system role. For those
    templates, retry with the system instruction prepended to the user message.
    """

    def apply(messages):
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors=return_tensors,
            continue_final_message=True,
        )
        if isinstance(rendered, dict):
            return rendered["input_ids"]
        try:
            return rendered["input_ids"]
        except (IndexError, KeyError, TypeError):
            return rendered

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant_prefill},
    ]
    try:
        return apply(messages)
    except Exception:
        fallback_messages = [
            {"role": "user", "content": f"{system}\n\n{user}"},
            {"role": "assistant", "content": assistant_prefill},
        ]
        try:
            return apply(fallback_messages)
        except Exception as fallback_error:
            raise RuntimeError(
                "Tokenizer does not provide a compatible chat template with assistant prefill support."
            ) from fallback_error
