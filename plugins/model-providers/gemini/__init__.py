"""Google Gemini provider profiles.

gemini:            Google AI Studio (API key) — uses GeminiNativeClient
google-gemini-cli: Google Cloud Code Assist (OAuth) — uses GeminiCloudCodeClient

Both report api_mode="chat_completions" but use custom native clients
that bypass the standard OpenAI transport. The profile captures auth
and endpoint metadata for auth.py / runtime_provider.py migration, and
carries the thinking_config translation hook so the transport's profile
path produces the same extra_body shape the legacy flag path did.
"""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class GeminiProfile(ProviderProfile):
    """Gemini — translate reasoning_config to thinking_config in extra_body."""

    def build_extra_body(
        self, *, session_id: str | None = None, **context: Any
    ) -> dict[str, Any]:
        """Emit extra_body.thinking_config (native) or extra_body.extra_body.google.thinking_config
        (OpenAI-compat /openai subpath), mirroring the legacy path's behavior.
        """
        from agent.transports.chat_completions import (
            _build_gemini_thinking_config,
            _is_gemini_openai_compat_base_url,
            _snake_case_gemini_thinking_config,
        )

        model = context.get("model") or ""
        reasoning_config = context.get("reasoning_config")
        base_url = context.get("base_url") or self.base_url

        raw_thinking_config = _build_gemini_thinking_config(model, reasoning_config)
        if not raw_thinking_config:
            return {}

        body: dict[str, Any] = {}
        if self.name == "gemini" and _is_gemini_openai_compat_base_url(base_url):
            thinking_config = _snake_case_gemini_thinking_config(raw_thinking_config)
            if thinking_config:
                body["extra_body"] = {"google": {"thinking_config": thinking_config}}
        else:
            body["thinking_config"] = raw_thinking_config
        return body


gemini = GeminiProfile(
    name="gemini",
    aliases=("google", "google-gemini", "google-ai-studio"),
    api_mode="chat_completions",
    env_vars=("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta",
    auth_type="api_key",
    default_aux_model="gemini-3.5-flash",
    fallback_models=(
        "gemini-3.1-pro-preview",
        "gemini-3-pro-preview",
        "gemini-3.5-flash",
        "gemini-3.1-flash-lite-preview",
    ),
)

gemini_vertex = GeminiProfile(
    name="gemini-vertex",
    aliases=("google-vertex", "vertex-gemini", "vertex-ai-gemini"),
    display_name="Google Vertex AI Gemini",
    description="Gemini through Vertex AI service account authentication",
    signup_url="https://console.cloud.google.com/vertex-ai",
    api_mode="chat_completions",
    env_vars=("VERTEX_AI_CREDENTIALS_FILE",),
    base_url="https://aiplatform.googleapis.com/v1/projects/professor-do-iphone/locations/global/publishers/google",
    auth_type="service_account",
    default_aux_model="gemini-flash-lite-latest",
    fallback_models=(
        "gemini-flash-lite-latest",
        "gemini-3.1-pro-preview",
        "gemini-3-pro-preview",
        "gemini-3.5-flash",
        "gemini-3.1-flash-lite-preview",
    ),
)

google_gemini_cli = GeminiProfile(
    name="google-gemini-cli",
    aliases=("gemini-cli", "gemini-oauth"),
    api_mode="chat_completions",
    env_vars=(),  # OAuth — no API key
    base_url="cloudcode-pa://google",  # Cloud Code Assist internal scheme
    auth_type="oauth_external",
)

register_provider(gemini)
register_provider(gemini_vertex)
register_provider(google_gemini_cli)
