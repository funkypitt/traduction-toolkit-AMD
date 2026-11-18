"""
Backend LLM « claude-code » partagé par traduire.py, traduire-pro.py, doubler.py,
doubler-mp3-batch.py, clipper.py, resumer.py et sous_titrer_docx.py.

Chaque appel passe par `claude -p` (Claude Code en mode non interactif) et donc
par l'abonnement Claude de la session, sans clé API ni crédits API.
Le client mime l'interface Anthropic utilisée par les scripts :
`client.messages.create(model=, max_tokens=, system=, messages=, tools=, tool_choice=)`
et renvoie un objet dont `.content` est une liste de blocs `text` ou `tool_use`.

- ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN sont retirées de l'environnement du
  sous-processus, sinon la CLI facturerait l'API.
- Le tool_use forcé (tool_choice = {"type": "tool"}) devient --json-schema ;
  le résultat revient dans `structured_output`.
- Aucun outil, aucun MCP, aucun réglage, répertoire de travail vide : aucun
  CLAUDE.md ni réglage de projet ne fuit dans le contexte.
- Les retries sont faits ici (les scripts ne retentent que les erreurs anthropic).

Choix du backend : option --llm claude-code. Pour en faire le défaut, la variable
TRADUCTION_LLM=claude-code, ou ce seul mot dans le fichier ~/.config/traduction/llm.
Le fichier vaut pour tout ce qui lance les scripts sans --llm, y compris ce qui
ne lit pas ~/.bashrc : gui.py ouvert depuis le bureau (qui s'en sert aussi pour
présélectionner sa liste) et le daemon de l'extension. La variable l'emporte sur
le fichier. Sans l'un ni l'autre, le défaut de chaque script ne change pas.
"""

import atexit
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

LLM_BACKENDS = ("local", "claude", "claude-code")
CONFIG_FILE = os.path.expanduser("~/.config/traduction/llm")

LLM_CHOICES_HELP = ("claude-code : `claude -p`, abonnement Claude, sans clé API. "
                    "Défaut modifiable par la variable TRADUCTION_LLM")
DEFAULT_MODEL = "opus"
RETRY_MAX = 4
RETRY_DELAY = 10.0
TIMEOUT_SEC = 900
ARG_MAX_BYTES = 100_000     # un argument de ligne de commande : 128 Ko au plus sous Linux


def default_llm(fallback: str = "claude") -> str:
    """Backend par défaut : la variable TRADUCTION_LLM, sinon le mot écrit dans
    ~/.config/traduction/llm, sinon `fallback`.

    Une valeur inconnue arrête le script : argparse ne contrôle pas ses défauts,
    et un repli muet lancerait un long traitement sur un autre backend que prévu.
    """
    value, origin = os.environ.get("TRADUCTION_LLM", "").strip(), "TRADUCTION_LLM"
    if not value:
        try:
            with open(CONFIG_FILE, encoding="utf-8") as f:
                value, origin = f.read().strip(), CONFIG_FILE
        except OSError:
            pass
    if not value:
        return fallback
    if value not in LLM_BACKENDS:
        sys.exit(f"❌ {origin} : valeur inconnue {value!r} "
                 f"(attendu : {', '.join(LLM_BACKENDS)})")
    return value


def find_bin():
    """Chemin du binaire `claude` : TRADUCTION_CLAUDE_BIN, le PATH, puis
    ~/.local/bin (absent du PATH d'un service systemd)."""
    cand = os.environ.get("TRADUCTION_CLAUDE_BIN") or shutil.which("claude")
    if cand:
        return cand
    local = os.path.expanduser("~/.local/bin/claude")
    return local if os.access(local, os.X_OK) else None


def analysis_client(llm: str, analysis_llm: str, client, cc_model: str = DEFAULT_MODEL):
    """Client de la passe d'analyse quand elle revient à Claude Code, sinon None
    (la règle habituelle du script s'applique alors).

    - --analysis-llm claude-code : toujours Claude Code, quel que soit --llm ;
    - --llm claude-code avec --analysis-llm auto : le même client ;
    - --llm local avec --analysis-llm auto : Claude Code si le binaire `claude`
      est là. L'abonnement passe avant l'API payante, même quand une
      ANTHROPIC_API_KEY est dans l'environnement ; l'API ne sert plus à
      l'analyse que demandée (--analysis-llm claude) ou faute de `claude`.
    """
    if analysis_llm == "claude-code":
        return client if isinstance(client, ClaudeCodeClient) else ClaudeCodeClient(model=cc_model)
    if analysis_llm == "auto":
        if llm == "claude-code":
            return client
        if llm == "local" and find_bin():
            return ClaudeCodeClient(model=cc_model)
    return None


def _via(default: str) -> str:
    via = os.environ.get("CLAUDE_VIA", "").strip() or default
    if via not in ("claude-code", "api"):
        sys.exit(f"❌ CLAUDE_VIA={via!r} : valeur inconnue (attendu : claude-code, api)")
    return via


def client(model: str = None, default: str = "claude-code", **anthropic_kwargs):
    """Client pour un script qui n'a pas d'option --llm : Claude Code par défaut
    (abonnement), l'API Anthropic si CLAUDE_VIA=api (ANTHROPIC_API_KEY requise).

    `model` vide : chaque appel garde le modèle qu'il demande (model=...).
    `default="api"` : pour un script qui règle la température, que `claude -p`
    ignore ; l'API reste alors son défaut et CLAUDE_VIA=claude-code l'option.
    Les autres arguments (api_key, max_retries…) ne servent qu'à l'API.
    """
    via = _via(default)
    if via == "api":
        import anthropic
        return anthropic.Anthropic(**anthropic_kwargs)
    return ClaudeCodeClient(model=model)


def api_key(default: str = "claude-code"):
    """Pour les contrôles « clé absente → on s'arrête » des scripts sans --llm :
    la vraie clé quand l'appel passe par l'API, un simple marqueur avec Claude
    Code (qui n'en demande pas). Même `default` que client()."""
    if _via(default) == "api":
        return os.environ.get("ANTHROPIC_API_KEY")
    return "claude-code"


class ClaudeCodeClient:
    """Client `claude -p` mimant anthropic.Anthropic (client.messages.create,
    client.messages.stream, client.beta.messages…).

    `model` fixe le modèle de tous les appels (alias ou ID) ; None laisse chaque
    appel choisir par son argument model=, comme avec l'API.
    Ce que `claude -p` ne sait pas rendre est refusé plutôt que perdu en
    silence : images et documents dans un message, tours « assistant ».
    temperature et max_tokens sont ignorés (la CLI ne les règle pas).
    """

    def __init__(self, model: str = DEFAULT_MODEL, effort: str = None):
        self.bin = find_bin()
        if not self.bin:
            raise RuntimeError("binaire `claude` introuvable (PATH ou TRADUCTION_CLAUDE_BIN)")
        self.model = model
        self.effort = effort
        self.cwd = tempfile.mkdtemp(prefix="vide-")   # nom neutre : la CLI le montre au modèle
        atexit.register(shutil.rmtree, self.cwd, True)
        self.messages = self            # client.messages.create → self.create
        self.beta = self                # client.beta.messages.stream → self.stream

    def __repr__(self):
        return f"ClaudeCodeClient({self.model})"

    @staticmethod
    def _prompt(messages) -> str:
        parts = []
        for m in messages:
            if m.get("role") != "user":
                raise NotImplementedError(
                    f"claude-code : message de rôle {m.get('role')!r} non pris en charge "
                    "(un seul tour « user ») — utiliser l'API pour cet appel")
            c = m["content"]
            if isinstance(c, str):
                parts.append(c)
                continue
            for b in c:
                kind = b.get("type") if isinstance(b, dict) else getattr(b, "type", None)
                if kind != "text":
                    raise NotImplementedError(
                        f"claude-code : bloc {kind!r} non pris en charge (texte seulement) "
                        "— utiliser l'API pour cet appel")
            parts.append("".join(b.get("text", "") for b in c))
        return "\n\n".join(parts)

    def _run(self, cmd, prompt):
        env = {k: v for k, v in os.environ.items()
               if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
        r = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                           cwd=self.cwd, env=env, timeout=TIMEOUT_SEC)
        try:
            data = json.loads(r.stdout)
        except json.JSONDecodeError:
            raise RuntimeError(f"claude -p a échoué (code {r.returncode}) : "
                               f"{(r.stderr or r.stdout)[-500:]}")
        if data.get("is_error"):
            raise RuntimeError(f"claude -p : {data.get('result') or data.get('subtype')}")
        return data

    def create(self, **kwargs):
        prompt = self._prompt(kwargs.get("messages", []))
        model = self.model or kwargs.get("model") or DEFAULT_MODEL
        cmd = [self.bin, "-p", "--model", model,
               "--tools", "", "--strict-mcp-config", "--setting-sources", "",
               "--no-session-persistence", "--output-format", "json"]
        effort = self.effort
        for source in (kwargs.get("output_config"), (kwargs.get("extra_body") or {}).get("output_config")):
            if isinstance(source, dict) and source.get("effort"):
                effort = source["effort"]
        if effort:
            cmd += ["--effort", effort]
        system_file = None
        if kwargs.get("system"):
            system = kwargs["system"]
            if not isinstance(system, str):
                system = "".join(b.get("text", "") for b in system if isinstance(b, dict))
            if len(system.encode("utf-8")) < ARG_MAX_BYTES:
                cmd += ["--system-prompt", system]
            else:
                # Linux refuse un argument de plus de 128 Ko : au-delà, par fichier.
                fd, system_file = tempfile.mkstemp(prefix="systeme-", suffix=".txt")
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(system)
                cmd += ["--system-prompt-file", system_file]
        tool = None
        choice = kwargs.get("tool_choice") or {}
        if choice.get("type") == "tool":
            tool = next(t for t in kwargs.get("tools", []) if t["name"] == choice["name"])
            cmd += ["--json-schema", json.dumps(tool["input_schema"])]

        try:
            for attempt in range(1, RETRY_MAX + 1):
                try:
                    data = self._run(cmd, prompt)
                    if tool is not None and not isinstance(data.get("structured_output"), dict):
                        raise RuntimeError("claude -p : sortie structurée absente")
                    break
                except (RuntimeError, subprocess.TimeoutExpired) as exc:
                    if attempt == RETRY_MAX:
                        raise
                    delay = RETRY_DELAY * (2 ** (attempt - 1))
                    print(f"   ⏳ Claude Code erreur ({str(exc)[:120]}), "
                          f"retry {attempt}/{RETRY_MAX} dans {delay:.0f}s...")
                    time.sleep(delay)
        finally:
            if system_file:
                os.unlink(system_file)

        if tool is not None:
            block = _Block(type="tool_use", name=tool["name"],
                           input=data["structured_output"])
        else:
            block = _Block(type="text", text=data.get("result", ""))
        u = data.get("usage") or {}
        usage = _Block(input_tokens=u.get("input_tokens", 0),
                       output_tokens=u.get("output_tokens", 0))
        # stop_reason vrai (« max_tokens » = réponse tronquée), pour qui le contrôle.
        return _Resp(content=[block], stop_reason=data.get("stop_reason") or "end_turn",
                     usage=usage)


    def stream(self, **kwargs):
        """client.messages.stream(...) : la réponse arrive d'un bloc, pas au fil de l'eau."""
        kwargs.pop("betas", None)
        return _Stream(self.create(**kwargs))


class _Stream:
    def __init__(self, message):
        self._message = message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @property
    def text_stream(self):
        for b in self._message.content:
            if b.type == "text":
                yield b.text

    def get_final_message(self):
        return self._message

    def get_final_text(self):
        return "".join(self.text_stream)


class _Block:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Resp:
    def __init__(self, **kw):
        self.__dict__.update(kw)
