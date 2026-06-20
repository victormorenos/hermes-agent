"""Natural gateway presentation for tool execution progress.

This module is deliberately UI-facing.  It turns internal tool calls into
short operational status text while keeping secrets, phone numbers, pairing
codes, sessions, and local paths out of auxiliary prompts and gateway logs.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Mapping, Optional


_ALLOWED_VISIBILITY = {"show", "silent", "media_only", "debug_only"}
_ALLOWED_MEDIA_POLICY = {"normal", "document", "voice", "none"}
_SENSITIVE_KEY_RE = re.compile(
    r"(token|secret|key|password|passwd|credential|cookie|authorization|"
    r"session|pairing|auth|api[_-]?key)",
    re.IGNORECASE,
)
_PHONE_RE = re.compile(r"\b\+?\d[\d\s().-]{8,}\d\b")
_PAIRING_RE = re.compile(r"\b[A-Z0-9]{6,12}\b")
_ABS_PATH_RE = re.compile(r"(?<![\w:/.-])(?:~|/Users|/private|/var|/tmp|/home|/root|/etc|/opt|/srv|/mnt)/[^\s\"'<>]+")
_WIN_PATH_RE = re.compile(r"\b[A-Za-z]:\\[^\s\"'<>]+")
_URL_SECRET_RE = re.compile(r"([?&](?:token|key|secret|code|session|auth)=[^&\s]+)", re.IGNORECASE)
_MEDIA_LINE_RE = re.compile(r"^\s*(?:\[\[(?:audio_as_voice|as_document)\]\]\s*)?MEDIA:.+$", re.MULTILINE)


@dataclass(frozen=True)
class ToolPresentation:
    display_thought: str = ""
    progress_label: str = ""
    visibility: str = "show"
    suppress_final_text: bool = False
    media_policy: str = "normal"

    def normalized(self) -> "ToolPresentation":
        visibility = self.visibility if self.visibility in _ALLOWED_VISIBILITY else "show"
        media_policy = self.media_policy if self.media_policy in _ALLOWED_MEDIA_POLICY else "normal"
        return ToolPresentation(
            display_thought=_clean_phrase(self.display_thought),
            progress_label=_clean_phrase(self.progress_label),
            visibility=visibility,
            suppress_final_text=bool(self.suppress_final_text),
            media_policy=media_policy,
        )


class ToolPresentationAgent:
    """Adapter from technical tool calls to concise Portuguese status text."""

    def present_tool_start(
        self,
        tool_name: Optional[str],
        args: Optional[Mapping[str, Any]] = None,
        *,
        platform: str = "",
        preview: Optional[str] = None,
        verbose: bool = False,
        tool_schema: Optional[Mapping[str, Any]] = None,
    ) -> ToolPresentation:
        name = _canonical_tool_name(tool_name)
        args = args or {}
        platform_key = str(platform or "").lower()

        if name in {"text_to_speech", "text_to_speech_tool"}:
            return ToolPresentation(
                visibility="silent",
                suppress_final_text=True,
                media_policy="voice",
            )

        if name in {"todo", "todo_tool"}:
            return self._present_todo(args)

        if name in {"image_generate", "image_generate_tool"}:
            return ToolPresentation(
                display_thought="Vou gerar uma imagem com esse pedido.",
                media_policy="document" if "whatsapp" in platform_key else "normal",
            )

        known = self._present_known(name, args, preview)
        if known:
            return known

        adapted = self._adapt_unknown_with_ai(
            name,
            args,
            platform=platform_key,
            preview=preview,
            verbose=verbose,
            tool_schema=tool_schema,
        )
        return adapted.normalized()

    def _present_known(
        self,
        name: str,
        args: Mapping[str, Any],
        preview: Optional[str],
    ) -> Optional[ToolPresentation]:
        if name in {"web_search", "search_web"}:
            return ToolPresentation(display_thought="Vou pesquisar isso rapidamente.")
        if name in {"web_extract", "fetch_url"}:
            return ToolPresentation(display_thought="Vou abrir a fonte e resumir o essencial.")
        if name in {"session_search", "memory_search", "recall"}:
            return ToolPresentation(display_thought="Vou procurar no histórico da conversa.")
        if name in {"read_file", "search_files", "list_files"}:
            return ToolPresentation(display_thought="Vou verificar os arquivos necessários.")
        if name in {"write_file", "patch", "apply_patch"}:
            return ToolPresentation(display_thought="Vou aplicar a alteração necessária.")
        if name in {"terminal", "execute_code", "process"}:
            return ToolPresentation(display_thought="Vou executar uma etapa técnica necessária.")
        if name in {"browser_navigate", "browser_click", "browser_type"}:
            return ToolPresentation(display_thought="Vou interagir com a página.")
        if name in {"vision_analyze", "image_analyze"}:
            return ToolPresentation(display_thought="Vou analisar a imagem.")
        if name in {"delegate_task", "mixture_of_agents"}:
            return ToolPresentation(display_thought="Vou dividir o trabalho e consolidar o resultado.")
        if name in {"skills_list", "skill_view", "skill_manage"}:
            return ToolPresentation(display_thought="Vou consultar as habilidades disponíveis.")
        if name in {"clarify"}:
            return ToolPresentation(visibility="silent")
        return None

    def _present_todo(self, args: Mapping[str, Any]) -> ToolPresentation:
        todos = args.get("todos")
        if todos is None:
            return ToolPresentation(display_thought="Vou consultar o plano atual.")
        if not isinstance(todos, list):
            return ToolPresentation(display_thought="Vou atualizar o plano de trabalho.")

        items = [item for item in todos if isinstance(item, Mapping)]
        total = len(items)
        if total <= 0:
            return ToolPresentation(display_thought="Vou atualizar o plano de trabalho.")

        merge = bool(args.get("merge"))
        if not merge:
            return ToolPresentation(
                display_thought=f"Vou elaborar um plano de {total} passos."
            )

        completed = sum(1 for item in items if _status(item) == "completed")
        active = next((item for item in items if _status(item) == "in_progress"), None)
        if active:
            step = min(total, completed + 1)
            content = _short_task_label(active.get("content") or active.get("id") or "etapa atual")
            return ToolPresentation(progress_label=f"{step}/{total} {content}...")
        if completed:
            return ToolPresentation(progress_label=f"{completed}/{total} Atualizando o plano...")
        return ToolPresentation(display_thought=f"Vou atualizar {total} passos do plano.")

    def _adapt_unknown_with_ai(
        self,
        name: str,
        args: Mapping[str, Any],
        *,
        platform: str,
        preview: Optional[str],
        verbose: bool,
        tool_schema: Optional[Mapping[str, Any]],
    ) -> ToolPresentation:
        payload = {
            "tool_name": _safe_tool_name(name),
            "platform": _sanitize_scalar(platform, 40),
            "preview": _sanitize_scalar(preview or "", 160),
            "args": sanitize_for_auxiliary(args),
            "schema": sanitize_for_auxiliary(_compact_schema(tool_schema)),
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "Voce adapta execucoes de tools em mensagens curtas de UX para gateway. "
                    "Responda apenas JSON valido com: display_thought, progress_label, "
                    "visibility, suppress_final_text, media_policy. Use portugues do Brasil. "
                    "Nunca exponha nomes tecnicos crus, segredos, telefones, sessoes, paths "
                    "completos ou argumentos sensiveis. display_thought deve ser intencao "
                    "operacional curta, nao chain-of-thought privado."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, sort_keys=True),
            },
        ]
        try:
            from agent.auxiliary_client import call_llm

            response = call_llm(
                task="tool_presentation",
                messages=messages,
                temperature=0.2,
                max_tokens=180,
                timeout=8,
            )
            content = _response_content(response)
            parsed = _parse_json_object(content)
            if parsed:
                return ToolPresentation(
                    display_thought=str(parsed.get("display_thought") or ""),
                    progress_label=str(parsed.get("progress_label") or ""),
                    visibility=str(parsed.get("visibility") or "show"),
                    suppress_final_text=bool(parsed.get("suppress_final_text")),
                    media_policy=str(parsed.get("media_policy") or "normal"),
                )
        except Exception:
            pass

        return ToolPresentation(display_thought="Vou executar uma etapa necessária.")


def format_progress_message(
    presentation: ToolPresentation,
    *,
    verbose: bool = False,
    technical_fallback: Optional[str] = None,
) -> Optional[str]:
    item = presentation.normalized()
    if item.visibility in {"silent", "media_only"}:
        return None
    if item.visibility == "debug_only" and not verbose:
        return None
    if item.progress_label:
        return item.progress_label
    if item.display_thought:
        return f"🧠 {item.display_thought}"
    if verbose and technical_fallback:
        return technical_fallback
    return None


def translate_approval_description(description: Any) -> str:
    """Translate known approval reasons into concise PT-BR text."""

    text = str(description or "").strip()
    lowered = text.lower()
    if "execute_code script execution" in lowered:
        return (
            "execucao de codigo. O script pode chamar subprocessos ou alterar "
            "arquivos sem passar pela aprovacao de comandos do terminal; a "
            "aprovacao vale apenas para esta execucao."
        )
    if lowered in {"dangerous command", "command requires approval"}:
        return "acao sensivel que precisa da sua confirmacao."
    return _clean_phrase(text) or "acao sensivel que precisa da sua confirmacao."


def format_exec_approval_prompt(
    command: Any,
    description: Any = "",
    *,
    command_prefix: str = "/",
    allow_permanent: bool = True,
    max_command_chars: int = 800,
) -> str:
    """Build the plain-text fallback for gateway approval prompts."""

    prefix = command_prefix or "/"
    cmd = str(command or "")
    preview = cmd[:max_command_chars] + "..." if len(cmd) > max_command_chars else cmd
    reason = translate_approval_description(description)
    options = [
        f"`{prefix}approve` aprovar uma vez",
        f"`{prefix}approve session` aprovar nesta sessao",
    ]
    if allow_permanent:
        options.append(f"`{prefix}approve always` aprovar sempre")
    options.append(f"`{prefix}deny` negar")
    return (
        "⚠️ **Preciso da sua aprovacao para continuar**\n"
        f"```\n{preview}\n```\n"
        f"Motivo: {reason}\n\n"
        "Escolha uma opcao:\n"
        + "\n".join(f"- {item}" for item in options)
    )


def approval_choice_command(choice: str) -> str:
    normalized = str(choice or "").strip().lower()
    if normalized in {"approve", "once", "one"}:
        return "/approve"
    if normalized in {"session", "sessao", "sessão"}:
        return "/approve session"
    if normalized in {"always", "permanent", "permanente"}:
        return "/approve always"
    if normalized in {"deny", "cancel", "negate", "negar"}:
        return "/deny"
    return ""


def approval_confirmation_text(choice: str, count: int = 1) -> str:
    normalized = str(choice or "").strip().lower()
    plural = count > 1
    if normalized in {"approve", "once", "one"}:
        if not plural:
            return "✅ Aprovado. O agente vai continuar..."
        return f"✅ Aprovados {count} comandos. O agente vai continuar..."
    if normalized in {"session", "sessao", "sessão"}:
        if not plural:
            return "✅ Aprovado para esta sessao. O agente vai continuar..."
        return f"✅ Aprovados {count} comandos para esta sessao. O agente vai continuar..."
    if normalized in {"always", "permanent", "permanente"}:
        if not plural:
            return "✅ Aprovado sempre. O agente vai continuar..."
        return f"✅ Aprovados {count} comandos permanentemente. O agente vai continuar..."
    return "❌ Negado." if not plural else f"❌ Negados {count} comandos."


def activity_label(action: Any) -> str:
    """Map technical activity names to short Portuguese progress labels."""

    raw = str(action or "").strip()
    if not raw:
        return ""
    name = _canonical_tool_name(raw.split("(", 1)[0].split()[0])
    mapping = {
        "execute_code": "executando uma etapa tecnica",
        "terminal": "executando uma etapa tecnica",
        "process": "acompanhando uma tarefa",
        "web_search": "pesquisando fontes",
        "search_web": "pesquisando fontes",
        "web_extract": "abrindo uma fonte",
        "fetch_url": "abrindo uma fonte",
        "text_to_speech": "gerando audio",
        "image_generate": "gerando imagem",
        "todo": "organizando o plano",
        "session_search": "consultando o historico",
        "memory_search": "consultando a memoria",
        "vision_analyze": "analisando imagem",
    }
    if name in mapping:
        return mapping[name]
    sanitized = _sanitize_scalar(raw, 80)
    if not sanitized:
        return ""
    if re.fullmatch(r"[a-zA-Z0-9_.:-]+", sanitized):
        return "executando uma etapa"
    return sanitized


def format_gateway_heartbeat(
    elapsed_mins: int,
    *,
    iteration: Optional[int] = None,
    max_iterations: Optional[int] = None,
    action: Any = "",
    include_detail: bool = True,
) -> str:
    parts: list[str] = []
    if include_detail and iteration and max_iterations:
        parts.append(f"etapa {iteration}/{max_iterations}")
    label = activity_label(action)
    if label:
        parts.append(label)
    suffix = f" — {', '.join(parts)}" if parts else ""
    return f"⏳ Trabalhando — {elapsed_mins} min{suffix}"


def sanitize_for_auxiliary(value: Any, *, max_depth: int = 3, max_items: int = 12) -> Any:
    return _sanitize_value(value, max_depth=max_depth, max_items=max_items)


def current_turn_has_successful_tts(
    messages: Iterable[Mapping[str, Any]],
    *,
    history_offset: int = 0,
) -> bool:
    return bool(_current_turn_successful_tts_texts(messages, history_offset=history_offset))


def suppress_tts_duplicate_text(
    response: str,
    messages: Iterable[Mapping[str, Any]],
    *,
    history_offset: int = 0,
) -> str:
    """Return a media-only response when TTS already delivered duplicate text."""

    if not response or "MEDIA:" not in response:
        return response
    text_lines, media_lines = _split_text_and_media_lines(response)
    if not media_lines:
        return response

    tts_texts = _current_turn_successful_tts_texts(messages, history_offset=history_offset)
    if not tts_texts:
        return response

    visible_text = _normalize_visible_text("\n".join(text_lines))
    if not visible_text:
        return "\n".join(media_lines)

    for tts_text in tts_texts:
        if _looks_like_tts_duplicate(visible_text, _normalize_visible_text(tts_text)):
            return "\n".join(media_lines)
    return response


def _current_turn_successful_tts_texts(
    messages: Iterable[Mapping[str, Any]],
    *,
    history_offset: int,
) -> List[str]:
    msg_list = list(messages or [])
    if history_offset and len(msg_list) >= history_offset:
        msg_list = msg_list[history_offset:]

    call_text: Dict[str, str] = {}
    for msg in msg_list:
        if msg.get("role") != "assistant":
            continue
        for call in msg.get("tool_calls") or []:
            fn = call.get("function") or {}
            name = _canonical_tool_name(fn.get("name") or call.get("name"))
            if name not in {"text_to_speech", "text_to_speech_tool"}:
                continue
            call_id = str(call.get("id") or call.get("call_id") or "")
            args_raw = fn.get("arguments") or call.get("arguments") or {}
            args = _loads_mapping(args_raw)
            text = str(args.get("text") or args.get("input") or "").strip()
            if call_id and text:
                call_text[call_id] = text

    successful: List[str] = []
    for msg in msg_list:
        if msg.get("role") not in {"tool", "function"}:
            continue
        call_id = str(msg.get("tool_call_id") or msg.get("call_id") or "")
        text = call_text.get(call_id)
        if not text:
            continue
        content = str(msg.get("content") or "")
        if "MEDIA:" not in content:
            continue
        parsed = _parse_json_object(content) or {}
        if parsed and parsed.get("success") is False:
            continue
        successful.append(text)
    return successful


def _looks_like_tts_duplicate(visible: str, spoken: str) -> bool:
    if not visible or not spoken:
        return False
    if len(visible) <= 140 and any(
        word in visible
        for word in ("audio", "consigo", "claro", "pronto", "segue", "aqui esta", "feito", "sim")
    ):
        return True
    if visible in spoken or spoken in visible:
        return True
    ratio = SequenceMatcher(None, visible, spoken).ratio()
    return ratio >= 0.72


def _split_text_and_media_lines(response: str) -> tuple[List[str], List[str]]:
    text_lines: List[str] = []
    media_lines: List[str] = []
    for line in response.splitlines():
        stripped = line.strip()
        if stripped in {"[[audio_as_voice]]", "[[as_document]]"} or stripped.startswith("MEDIA:"):
            media_lines.append(stripped)
        elif stripped:
            text_lines.append(line)
    return text_lines, media_lines


def _normalize_visible_text(text: str) -> str:
    text = _MEDIA_LINE_RE.sub("", text or "")
    text = text.replace("[[audio_as_voice]]", "").replace("[[as_document]]", "")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _canonical_tool_name(tool_name: Optional[str]) -> str:
    return str(tool_name or "").strip().lower()


def _safe_tool_name(name: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_.:-]+", "_", name or "tool")
    return name[:80] or "tool"


def _status(item: Mapping[str, Any]) -> str:
    return str(item.get("status") or "").strip().lower()


def _short_task_label(value: Any) -> str:
    text = _clean_phrase(_sanitize_scalar(value, 80))
    text = text.rstrip(".")
    return text or "Etapa em andamento"


def _clean_phrase(text: Any) -> str:
    value = str(text or "").strip()
    value = re.sub(r"\s+", " ", value)
    return value[:180].strip()


def _sanitize_value(value: Any, *, max_depth: int, max_items: int) -> Any:
    if max_depth <= 0:
        return "[resumido]"
    if isinstance(value, Mapping):
        clean: Dict[str, Any] = {}
        for idx, (key, item) in enumerate(value.items()):
            if idx >= max_items:
                clean["..."] = f"{len(value) - max_items} item(s) omitidos"
                break
            key_str = str(key)
            if _SENSITIVE_KEY_RE.search(key_str):
                clean[key_str[:80]] = "[redigido]"
            else:
                clean[key_str[:80]] = _sanitize_value(
                    item,
                    max_depth=max_depth - 1,
                    max_items=max_items,
                )
        return clean
    if isinstance(value, (list, tuple, set)):
        seq = list(value)
        clean_list = [
            _sanitize_value(item, max_depth=max_depth - 1, max_items=max_items)
            for item in seq[:max_items]
        ]
        if len(seq) > max_items:
            clean_list.append(f"{len(seq) - max_items} item(s) omitidos")
        return clean_list
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return _sanitize_scalar(value, 240)


def _sanitize_scalar(value: Any, max_len: int) -> str:
    text = str(value or "")
    text = _URL_SECRET_RE.sub("?redigido=[redigido]", text)
    text = re.sub(
        r"MEDIA:(?:~|/Users|/private|/var|/tmp|/home|/root|/etc|/opt|/srv|/mnt)/[^\s\"'<>]+",
        "MEDIA:[arquivo]",
        text,
    )
    text = _ABS_PATH_RE.sub("[arquivo]", text)
    text = _WIN_PATH_RE.sub("[arquivo]", text)
    text = _PHONE_RE.sub("[numero]", text)
    text = _PAIRING_RE.sub(lambda m: "[codigo]" if any(ch.isdigit() for ch in m.group(0)) else m.group(0), text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_len:
        text = text[: max(0, max_len - 3)].rstrip() + "..."
    return text


def _compact_schema(tool_schema: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not isinstance(tool_schema, Mapping):
        return {}
    fn = tool_schema.get("function") if isinstance(tool_schema.get("function"), Mapping) else tool_schema
    return {
        "name": fn.get("name"),
        "description": fn.get("description"),
        "parameters": fn.get("parameters"),
    }


def _response_content(response: Any) -> str:
    try:
        return str(response.choices[0].message.content or "")
    except Exception:
        return str(response or "")


def _parse_json_object(raw: Any) -> Optional[Dict[str, Any]]:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _loads_mapping(raw: Any) -> Mapping[str, Any]:
    if isinstance(raw, Mapping):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, Mapping) else {}
        except Exception:
            return {}
    return {}
