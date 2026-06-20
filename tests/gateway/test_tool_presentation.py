import json
from types import SimpleNamespace

from gateway.tool_presentation import (
    ToolPresentationAgent,
    format_progress_message,
    sanitize_for_auxiliary,
    suppress_tts_duplicate_text,
)


def test_tts_progress_is_silent_and_voice_policy():
    item = ToolPresentationAgent().present_tool_start(
        "text_to_speech",
        {"text": "Consigo sim, Victor!"},
        platform="whatsapp",
    )

    assert item.visibility == "silent"
    assert item.suppress_final_text is True
    assert item.media_policy == "voice"
    assert format_progress_message(item) is None


def test_image_generation_uses_natural_thought_and_document_on_whatsapp():
    item = ToolPresentationAgent().present_tool_start(
        "image_generate",
        {"prompt": "um gato fofinho"},
        platform="whatsapp",
    )

    assert item.media_policy == "document"
    message = format_progress_message(item)
    assert message == "🧠 Vou gerar uma imagem com esse pedido."
    assert "image_generate" not in message


def test_todo_create_plan_becomes_visible_mission():
    todos = [
        {"id": str(idx), "content": f"Etapa {idx}", "status": "pending"}
        for idx in range(1, 6)
    ]

    item = ToolPresentationAgent().present_tool_start(
        "todo",
        {"todos": todos, "merge": False},
        platform="whatsapp",
    )

    assert format_progress_message(item) == "🧠 Vou elaborar um plano de 5 passos."


def test_todo_update_progress_label_uses_step_count():
    todos = [
        {"id": "1", "content": "Entender pedido", "status": "completed"},
        {"id": "2", "content": "Editar gateway", "status": "completed"},
        {"id": "3", "content": "Verificando configuração de áudio", "status": "in_progress"},
        {"id": "4", "content": "Rodar testes", "status": "pending"},
        {"id": "5", "content": "Publicar", "status": "pending"},
    ]

    item = ToolPresentationAgent().present_tool_start(
        "todo",
        {"todos": todos, "merge": True},
        platform="whatsapp",
    )

    assert format_progress_message(item) == "3/5 Verificando configuração de áudio..."


def test_unknown_tool_uses_auxiliary_adapter_with_sanitized_prompt(monkeypatch):
    captured = {}

    def fake_call_llm(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "display_thought": "Vou organizar essa etapa.",
                                "progress_label": "",
                                "visibility": "show",
                                "suppress_final_text": False,
                                "media_policy": "normal",
                            }
                        )
                    )
                )
            ]
        )

    monkeypatch.setattr("agent.auxiliary_client.call_llm", fake_call_llm)

    item = ToolPresentationAgent().present_tool_start(
        "future_secret_tool",
        {
            "api_key": "sk-live-should-not-leak",
            "phone": "5511936195058",
            "path": "/Users/victor/.hermes/whatsapp/session.json",
            "prompt": "gerar arquivo seguro",
        },
        platform="whatsapp",
    )

    prompt = json.dumps(captured["messages"], ensure_ascii=False)
    assert "sk-live-should-not-leak" not in prompt
    assert "5511936195058" not in prompt
    assert "/Users/victor/.hermes/whatsapp/session.json" not in prompt
    assert "[redigido]" in prompt
    assert "[numero]" in prompt
    assert "[arquivo]" in prompt
    assert format_progress_message(item) == "🧠 Vou organizar essa etapa."


def test_unknown_tool_fallback_is_safe_when_auxiliary_fails(monkeypatch):
    def fail_call_llm(**kwargs):
        raise RuntimeError("no aux")

    monkeypatch.setattr("agent.auxiliary_client.call_llm", fail_call_llm)

    item = ToolPresentationAgent().present_tool_start(
        "future_raw_tool",
        {"payload": {"token": "secret"}},
        platform="whatsapp",
    )

    assert format_progress_message(item) == "🧠 Vou executar uma etapa necessária."


def test_sanitizer_redacts_sensitive_shapes():
    sanitized = sanitize_for_auxiliary(
        {
            "token": "abc123",
            "text": "telefone 5511936195058 arquivo /tmp/hermes/session.json MEDIA:/tmp/audio.ogg",
            "nested": {"pairing_code": "8REBP24V"},
        }
    )

    rendered = json.dumps(sanitized, ensure_ascii=False)
    assert "abc123" not in rendered
    assert "5511936195058" not in rendered
    assert "/tmp/hermes/session.json" not in rendered
    assert "/tmp/audio.ogg" not in rendered
    assert "8REBP24V" not in rendered


def test_tts_success_suppresses_duplicate_final_text_but_keeps_audio():
    response = "Consigo sim 🙂\n[[audio_as_voice]]\nMEDIA:/tmp/voice.ogg"
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_tts",
                    "function": {
                        "name": "text_to_speech",
                        "arguments": json.dumps(
                            {"text": "Consigo sim, Victor! Aqui está um áudio curto."}
                        ),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_tts",
            "content": json.dumps(
                {
                    "success": True,
                    "media_tag": "[[audio_as_voice]]\nMEDIA:/tmp/voice.ogg",
                }
            ),
        },
    ]

    assert suppress_tts_duplicate_text(response, messages) == (
        "[[audio_as_voice]]\nMEDIA:/tmp/voice.ogg"
    )
