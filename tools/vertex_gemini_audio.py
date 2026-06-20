"""Helpers for Gemini audio calls through Vertex AI service accounts."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import quote
import uuid

import requests

DEFAULT_VERTEX_PROJECT = "professor-do-iphone"
DEFAULT_VERTEX_LOCATION = "global"
DEFAULT_VERTEX_CREDENTIALS_FILE = (
    "/www/wwwroot/hermes.atalh.us/data/hermes-home/credentials/vertex-ai-image-sa.json"
)
VERTEX_INLINE_AUDIO_MAX_BYTES = 20 * 1024 * 1024
DEFAULT_GCS_BUCKET = "hermes-ai"
DEFAULT_GCS_PREFIX = "hermes/audio"

_AUDIO_MIME_BY_EXT = {
    ".aac": "audio/aac",
    ".aif": "audio/aiff",
    ".aiff": "audio/aiff",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".mp3": "audio/mp3",
    ".mp4": "audio/mp4",
    ".mpeg": "audio/mpeg",
    ".mpga": "audio/mpeg",
    ".oga": "audio/ogg",
    ".ogg": "audio/ogg",
    ".opus": "audio/ogg",
    ".wav": "audio/wav",
    ".webm": "audio/webm",
}


def cfg_value(config: Dict[str, Any], *keys: str, default: str = "") -> str:
    for key in keys:
        value = config.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return default


def should_use_vertex(config: Dict[str, Any]) -> bool:
    auth = cfg_value(config, "auth", "mode", default="").lower()
    return auth in {"vertex", "vertex_ai", "vertex-ai", "service_account"}


def credentials_file(config: Dict[str, Any]) -> str:
    return (
        cfg_value(config, "credentials_file", "service_account_file")
        or os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
        or os.getenv("VERTEX_AI_CREDENTIALS_FILE", "").strip()
        or DEFAULT_VERTEX_CREDENTIALS_FILE
    )


def project(config: Dict[str, Any]) -> str:
    return (
        cfg_value(config, "project", "project_id")
        or os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
        or os.getenv("GOOGLE_CLOUD_PROJECT_ID", "").strip()
        or DEFAULT_VERTEX_PROJECT
    )


def location(config: Dict[str, Any]) -> str:
    return (
        cfg_value(config, "location", "region")
        or os.getenv("GOOGLE_CLOUD_LOCATION", "").strip()
        or os.getenv("GOOGLE_CLOUD_REGION", "").strip()
        or DEFAULT_VERTEX_LOCATION
    )


def endpoint(config: Dict[str, Any], model: str, method: str = "generateContent") -> str:
    loc = location(config)
    host = "aiplatform.googleapis.com" if loc == "global" else f"{loc}-aiplatform.googleapis.com"
    return (
        f"https://{host}/v1/projects/{project(config)}"
        f"/locations/{loc}/publishers/google/models/{model}:{method}"
    )


def native_base_url(config: Dict[str, Any]) -> str:
    loc = location(config)
    host = "aiplatform.googleapis.com" if loc == "global" else f"{loc}-aiplatform.googleapis.com"
    return f"https://{host}/v1/projects/{project(config)}/locations/{loc}/publishers/google"


def access_token(config: Dict[str, Any]) -> str:
    path = credentials_file(config)
    if not path or not Path(path).is_file():
        raise ValueError("Arquivo da service account do Vertex AI nao encontrado.")
    from google.auth.transport.requests import Request
    from google.oauth2 import service_account

    credentials = service_account.Credentials.from_service_account_file(
        path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    credentials.refresh(Request())
    return credentials.token


def audio_mime_type(file_path: str) -> str:
    path = Path(file_path)
    mime_type = _AUDIO_MIME_BY_EXT.get(path.suffix.lower())
    if not mime_type:
        guessed, _ = mimetypes.guess_type(str(path))
        mime_type = guessed or "application/octet-stream"
    return mime_type


def gcs_bucket(config: Dict[str, Any]) -> str:
    bucket = (
        cfg_value(config, "bucket", "gcs_bucket")
        or os.getenv("HERMES_GEMINI_GCS_BUCKET", "").strip()
        or DEFAULT_GCS_BUCKET
    )
    return bucket.removeprefix("gs://").strip("/")


def gcs_object_prefix(config: Dict[str, Any]) -> str:
    return (cfg_value(config, "object_prefix", "gcs_prefix", default=DEFAULT_GCS_PREFIX) or DEFAULT_GCS_PREFIX).strip("/")


def upload_audio_to_gcs(
    config: Dict[str, Any],
    file_path: str,
    *,
    mime_type: Optional[str] = None,
) -> str:
    path = Path(file_path)
    if not path.is_file():
        raise ValueError("Arquivo de audio nao encontrado para upload.")

    bucket = gcs_bucket(config)
    if not bucket:
        raise ValueError("Bucket GCS do Gemini nao configurado.")

    date_path = datetime.now(timezone.utc).strftime("%Y/%m/%d")
    suffix = path.suffix.lower() or ".audio"
    object_name = f"{gcs_object_prefix(config)}/{date_path}/{uuid.uuid4().hex}{suffix}"
    mime = mime_type or audio_mime_type(str(path))

    with path.open("rb") as body:
        response = requests.post(
            f"https://storage.googleapis.com/upload/storage/v1/b/{bucket}/o",
            params={"uploadType": "media", "name": object_name},
            headers={
                "Authorization": f"Bearer {access_token(config)}",
                "Content-Type": mime,
            },
            data=body,
            timeout=float(config.get("gcs_timeout", 120) or 120),
        )
    if response.status_code not in {200, 201}:
        try:
            err = response.json().get("error", {})
            detail = err.get("message") or response.text[:300]
        except Exception:
            detail = response.text[:300]
        raise RuntimeError(f"Falha ao enviar audio para GCS (HTTP {response.status_code}): {detail}")
    return f"gs://{bucket}/{object_name}"


def delete_gcs_object(config: Dict[str, Any], gs_uri: str) -> bool:
    if not isinstance(gs_uri, str) or not gs_uri.startswith("gs://"):
        return False
    bucket_and_name = gs_uri[5:]
    if "/" not in bucket_and_name:
        return False
    bucket, object_name = bucket_and_name.split("/", 1)
    if not bucket or not object_name:
        return False
    response = requests.delete(
        f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{quote(object_name, safe='')}",
        headers={"Authorization": f"Bearer {access_token(config)}"},
        timeout=float(config.get("gcs_timeout", 120) or 120),
    )
    return response.status_code in {200, 204, 404}


def post_generate_content(
    config: Dict[str, Any],
    model: str,
    payload: Dict[str, Any],
    *,
    timeout: float = 120,
) -> Dict[str, Any]:
    response = requests.post(
        endpoint(config, model, "generateContent"),
        headers={
            "Authorization": f"Bearer {access_token(config)}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    if response.status_code != 200:
        try:
            err = response.json().get("error", {})
            detail = err.get("message") or response.text[:300]
        except Exception:
            detail = response.text[:300]
        raise RuntimeError(f"Vertex Gemini API error (HTTP {response.status_code}): {detail}")
    return response.json()


def inline_audio_part(file_path: str) -> Dict[str, Any]:
    path = Path(file_path)
    size = path.stat().st_size
    if size > VERTEX_INLINE_AUDIO_MAX_BYTES:
        raise ValueError("Audio maior que 20 MB; reduza o arquivo antes de enviar ao Gemini.")
    mime_type = audio_mime_type(str(path))
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"inlineData": {"mimeType": mime_type, "data": data}}


def extract_text(data: Dict[str, Any]) -> str:
    candidates = data.get("candidates") if isinstance(data, dict) else None
    if not candidates:
        return ""
    content = candidates[0].get("content") if isinstance(candidates[0], dict) else None
    parts = content.get("parts", []) if isinstance(content, dict) else []
    texts = []
    for part in parts:
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            texts.append(part["text"])
    return "\n".join(texts).strip()


def extract_inline_audio_b64(data: Dict[str, Any]) -> Optional[str]:
    candidates = data.get("candidates") if isinstance(data, dict) else None
    if not candidates:
        return None
    content = candidates[0].get("content") if isinstance(candidates[0], dict) else None
    parts = content.get("parts", []) if isinstance(content, dict) else []
    for part in parts:
        if not isinstance(part, dict) or part.get("thought"):
            continue
        inline = part.get("inlineData") or part.get("inline_data")
        if isinstance(inline, dict) and "data" in inline:
            data = inline.get("data")
            return data if isinstance(data, str) else None
    return None
