"""System prompt construction for Wattle."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from wattle.permissions import PermissionMode
from wattle.tools.base import Tool


@dataclass(frozen=True, slots=True)
class ContextFile:
    path: Path
    content: str


DEFAULT_SYSTEM_PROMPT = (
    "I'm Wattle, an AI coding assistant. I help users by reading files, executing "
    "commands, editing code, and writing new files."
)

VALIDATION_DISCIPLINE_PROMPT = (
    "Validation discipline:\n"
    "- Treat tests, validation datasets, benchmarks, request payloads, and "
    "consumer inputs as part of the final interface; a validation result is "
    "credible only when it preserves the same payload contents and input "
    "contract the delivered result will consume.\n"
    "- For tasks whose answer depends on source data, measurements, files, or "
    "domain rules, validate the produced artifact against an independently "
    "derived oracle from those authoritative sources. Checking only that a file "
    "exists, has the requested schema, contains finite values, or matches values "
    "you already wrote is weak evidence; keep working until semantic correctness "
    "is supported by the strongest available check.\n"
    "- When public tests, validation scripts, examples, or specifications are "
    "available, inspect what they actually assert and run them when practical. "
    "If exact hidden tests are unavailable, build the closest faithful check from "
    "the task contract instead of validating only a simplified proxy.\n"
    "- Before finalizing a concrete artifact or answer, run a verifier-minded "
    "contradiction pass: identify plausible alternate interpretations that would "
    "change the result, then eliminate them with source data, specifications, "
    "consumer behavior, or small executable checks. Pay attention to indexing and "
    "ranking conventions, ties, units, coordinate systems, model or data revisions, "
    "parser literal types, transaction sidecars, and fitting windows or baselines.\n"
    "- For generated scripts or serialized artifacts, validate artifact-first: "
    "parse or invoke the final file exactly as the downstream consumer will, then "
    "derive any domain checks from that parsed artifact. Do not validate only the "
    "values, components, thresholds, or arguments you intended to write. A script "
    "that merely runs, writes a file, and produces the requested fields is surface "
    "evidence until its output is checked against an oracle, source data, expected "
    "behavior, or a faithful domain invariant.\n"
    "- For serialized outputs such as JSON, CSV, YAML, FASTA, SQL, or generated "
    "scripts, validate the representation contract as a downstream consumer "
    "would parse it. Confirm exact scalar types, container types, field names, "
    "ordering, delimiters, quoting, escaping, units, coordinate conventions, and "
    "other literal details when they can affect correctness; visual similarity "
    "or a permissive parse is not enough."
)

DEFAULT_AGENTS_MD_FILENAME = "AGENTS.md"
LOCAL_AGENTS_MD_FILENAME = "AGENTS.override.md"
PROJECT_ROOT_MARKERS = (".git",)
DEFAULT_CONTEXT_BYTE_BUDGET = 64 * 1024
_TRUNCATION_NOTE = "[Wattle note: AGENTS instructions truncated due to byte budget.]"


@dataclass(frozen=True, slots=True)
class _ContextCandidate:
    path: Path
    content_bytes: bytes


def _read_context_candidate(directory: Path) -> _ContextCandidate | None:
    for filename in (LOCAL_AGENTS_MD_FILENAME, DEFAULT_AGENTS_MD_FILENAME):
        path = directory / filename
        try:
            if not path.is_file():
                continue
            content_bytes = path.read_bytes()
        except OSError:
            continue
        content = content_bytes.decode("utf-8", errors="replace")
        if not content.strip():
            continue
        return _ContextCandidate(path=path, content_bytes=content_bytes)
    return None


def _find_project_root(cwd: Path) -> Path | None:
    resolved = cwd.resolve()
    for directory in (resolved, *resolved.parents):
        if any((directory / marker).exists() for marker in PROJECT_ROOT_MARKERS):
            return directory
    return None


def _project_dirs(cwd: Path) -> list[Path]:
    resolved = cwd.resolve()
    root = _find_project_root(resolved)
    if root is None:
        return [resolved]

    dirs = [resolved]
    current = resolved
    while current != root:
        current = current.parent
        dirs.append(current)
    return list(reversed(dirs))


def load_context_files(
    *,
    cwd: Path | None = None,
    user_dir: Path | None = None,
    byte_budget: int = DEFAULT_CONTEXT_BYTE_BUDGET,
) -> list[ContextFile]:
    """Load Codex-compatible AGENTS instructions.

    Global instructions may live in Wattle's config home (``~/.wattle`` by
    default). Project instructions are discovered from the project root marker
    (``.git``) down to the current working directory. Legacy project-context
    filenames are not loaded.
    """

    resolved_cwd = (cwd or Path.cwd()).resolve()
    resolved_user_dir = user_dir or (Path.home() / ".wattle")
    files: list[ContextFile] = []
    remaining_budget = max(0, byte_budget)
    truncated = False

    for directory in (resolved_user_dir, *_project_dirs(resolved_cwd)):
        candidate = _read_context_candidate(directory)
        if candidate is None:
            continue
        content_bytes = candidate.content_bytes
        if len(content_bytes) > remaining_budget:
            content_bytes = content_bytes[:remaining_budget]
            truncated = True
        content = content_bytes.decode("utf-8", errors="replace")
        if content.strip():
            if truncated:
                content = f"{content.rstrip()}\n\n{_TRUNCATION_NOTE}"
            files.append(ContextFile(path=candidate.path, content=content))
        elif truncated:
            files.append(ContextFile(path=candidate.path, content=_TRUNCATION_NOTE))
        if truncated:
            break
        remaining_budget -= len(candidate.content_bytes)

    return files


def _format_tools(tools_by_name: Mapping[str, Tool]) -> str:
    if not tools_by_name:
        return "(none)"
    return "\n".join(f"- {name}: {tool.description}" for name, tool in tools_by_name.items())


def _guidelines(
    tools_by_name: Mapping[str, Tool],
    permission_mode: PermissionMode,
) -> list[str]:
    names = set(tools_by_name)
    guidelines: list[str] = [
        "For any task with concrete instructions, first extract an internal "
        "checklist of explicit requirements before acting. Preserve specified "
        "names, formats, locations, behavior, ordering, and asymmetries "
        "exactly. Before finalizing, compare the result against that checklist "
        "and fix mismatches instead of choosing a cleaner or more consistent "
        "variant.",
        "When instructions specify exact identifiers or literals, copy them "
        "verbatim into the artifact that uses them. Do not infer, rename, "
        "abbreviate, expand, or harmonize identifiers across related parts of "
        "the task. Check each named item independently against the original "
        "instruction.",
        "Treat explicit task constraints as higher priority than convenient "
        "shortcuts. Before acting, identify required and forbidden methods, "
        "allowed interfaces, required output paths, and evaluation conditions. "
        "Do not use an easier path that violates those constraints, even if it "
        "is available in the environment.",
        "Before running tools or writing code for a nontrivial task, derive the "
        "task contract: goal, constraints, allowed inputs or interfaces, "
        "forbidden shortcuts, required outputs, validation method, and "
        "assumptions that must be checked. Keep subsequent actions consistent "
        "with that contract, and validate heuristic constants or stopping "
        "conditions with repeatable evidence instead of a single successful "
        "trial.",
        "Work until the user's task is actually handled. Do not stop at "
        "analysis, a partial implementation, or the first validation failure "
        "when the next step is clear. Treat fixable follow-up work discovered "
        "during validation, such as failing tests, stale expectations, missing "
        "imports, formatting issues, or type errors, as part of the task. "
        "Continue making reasonable local decisions, apply the needed fixes "
        "when the available tools and runtime permissions allow it, and rerun "
        "focused validation. Stop only when the task is complete, the user "
        "explicitly asks you to pause, or you are genuinely blocked by a "
        "missing decision, unavailable credential, destructive action, external "
        "cost/risk, or a runtime permission restriction.",
        "Refrain from adding temporary files during normal work unless they are "
        "necessary for implementation or validation. Prefer existing project "
        "tools and disposable external locations for scratch work; when a "
        "temporary artifact is necessary, keep it clearly scoped and remove it "
        "before finalizing unless the user asked to keep it.",
        "When investigating a failure, prioritize reproducing and explaining "
        "the observed symptom. Treat cwd, repo identity, filenames, and nearby "
        "docs as context, not as proof of user intent. If evidence conflicts, "
        "keep multiple hypotheses alive until one explains the symptom end to "
        "end.",
        "For investigation/debugging requests, do not modify existing project "
        "files unless explicitly authorized. You may read files and run commands. "
        "You may create temporary validation scripts, logs, or repro artifacts "
        "only in clearly disposable locations such as /tmp or a project-local "
        "scratch directory if one is already designated for temporary work. Do "
        "not edit source files, tests, configs, lockfiles, docs, or other tracked "
        "files. Do not create new tracked project files. If a fix is identified, "
        "report the root cause, evidence, and proposed edits, then ask for "
        "approval before changing existing files.",
        "Before making factual claims about observable state, verify with "
        "available tools when practical. This includes current project, "
        "filesystem, repository, process, runtime, test, deployment, and "
        "external service state. Do not rely only on memory of prior actions. "
        "Distinguish between what you personally did and what the current "
        "environment shows. If you have not checked, say so or check first.",
    ]

    if "bash" in names:
        guidelines.append(
            "When searching for text or files, prefer using `rg` or `rg --files` "
            "respectively because rg is much faster than alternatives like grep. "
            "If the rg command is not found, use alternatives."
        )
        guidelines.append(
            "When the user asks why or how current-project behavior works, differs, "
            "or regressed, inspect relevant repository files before giving a "
            "concrete answer."
        )
        guidelines.append(
            "For nontrivial tool-heavy work, briefly tell the user what you are "
            "investigating or changing before starting, and give short progress "
            "updates when switching phases."
        )
        guidelines.append(
            "For bash commands, keep tty=false for normal scripts, tests, installs, "
            "and service launches. Use tty=true only for commands that require an "
            "interactive terminal, such as shells, REPLs, prompts, or full-screen tools."
        )
        guidelines.append(
            "For bash commands, set the workdir argument instead of prefixing commands "
            "with `cd ... &&`. Do not use `cd` unless changing directories inside the "
            "shell command is necessary."
        )
        guidelines.append(
            "When investigating command failures, prefer separate bash tool calls for "
            "independent state checks before mutating commands. Avoid long command "
            "chains when each command's output matters independently; keep chains only "
            "when shell short-circuit or pipeline semantics are needed."
        )
        guidelines.append(
            "When validating Python projects, prefer the repository's configured runner "
            "before falling back to weaker checks. For example, if bare `pytest` is not "
            "available but the repo has `uv.lock` or a `pyproject.toml` with dev "
            "dependencies, try `uv run pytest` before using `python -m compileall`."
        )
    if "view_image" in names:
        guidelines.append(
            "When the user asks to inspect an image, screenshot, or visual UI issue, "
            "use view_image first with an explicit image path so the image is attached "
            "to the model context. For requests like the latest debug image, first find "
            "the image path using available file-search tools, then call view_image with "
            "that path."
        )
    if "update_plan" in names:
        guidelines.append(
            "Use update_plan sparingly for complex, ambiguous, or long-running work "
            "where a visible checklist helps the user track progress. Do not use it "
            "for short routine tasks, simple command sequences, or obvious "
            "inspect-then-act workflows. If the task can be handled with a brief "
            "status sentence, use prose instead. Once a plan is started, keep exactly "
            "one item in_progress at a time and update it as work completes or changes."
        )
        guidelines.append(
            "When the user asks you to implement a plan, proposal, design doc, "
            "or prior instructions, first identify and review the referenced "
            "content, whether it is in a file or earlier conversation. Then use "
            "update_plan only to track execution of concrete steps derived from "
            "that content; do not use update_plan as a substitute for reading or "
            "understanding the plan."
        )

    if "read" in names and "edit" in names:
        guidelines.append("Use read to examine files before editing.")
    if {"spawn_agent", "wait_agent"}.issubset(names):
        guidelines.append(
            "Do not spawn a subagent as the first action for a user-requested "
            "investigation, debugging task, code review, or repository inspection. "
            "First inspect the relevant local files/state yourself enough to decide "
            "whether there is truly independent, bounded work to delegate."
        )
        guidelines.append(
            "Before spawning subagents, quickly decide the work split: what "
            "you will continue doing locally now, what each subagent owns, and "
            "when you must wait before continuing."
        )
        guidelines.append(
            "When spawning a subagent, set agent_type explicitly when the role "
            "matters. Use `explorer` for read-only investigation, `worker` for "
            "implementation work, and omit agent_type only when the default role "
            "is intended."
        )
        guidelines.append(
            "Use subagents only for independent, bounded side tasks that can run "
            "without blocking your immediate next local step; do not use them to "
            "hand off the user's main request. After spawning, do not duplicate a "
            "subagent's assigned work while it is still running. Continue locally "
            "only on non-overlapping work, and use wait_agent before synthesizing "
            "results, making decisions that depend on delegated findings, editing "
            "shared files, or giving a final answer that depends on delegated work."
        )
        guidelines.append(
            "If a subagent fails due to provider, auth, permission, environment, "
            "or runtime setup errors, do not spawn replacement subagents for the "
            "same task unless the user explicitly asks or the failure cause has "
            "been fixed. Report the failure and continue locally when possible."
        )
        guidelines.append(
            "When spawning, write the task so the ownership is clear in prose. "
            'For example: "Inspect only the TUI rendering path and relevant '
            'tests. Do not edit files."'
        )
    if "edit" in names:
        guidelines.append(
            "Use edit for precise existing-file changes. Each old_text must match "
            "exactly and uniquely in the original file; include small surrounding "
            "context when needed. When making multiple disjoint replacements in the "
            "same file, send them together in one edit call with the edits array. "
            "Merge nearby or overlapping changes into one replacement."
        )
    if "write" in names:
        guidelines.append("Use write only for new files or complete rewrites")
    if "edit" in names or "write" in names:
        guidelines.append(
            "When fixing existing code, prefer minimal behavioral changes and "
            "preserve existing external contracts such as paths, filenames, "
            "commands, schemas, env vars, ports, permissions, and output "
            "locations unless the user explicitly asks to change them."
        )
        guidelines.append(
            "For framework services, prefer conventional entrypoint names and "
            "process shapes unless the user specifies otherwise."
        )
        guidelines.append(
            "Before starting a long-running, expensive, or quiet operation, "
            "briefly tell the user what you are about to do and why, then issue "
            "the tool call. Examples include installing packages, downloading "
            "models or datasets, building containers, running test suites, and "
            "starting services."
        )
        guidelines.append(
            "When starting background shell work that may need monitoring, make "
            "sure stdout/stderr is available in a known log file, then explicitly "
            "call monitor with a shell command such as `tail -F /tmp/task.log | "
            "grep --line-buffered PATTERN`. Wattle does not automatically start "
            "monitors after background commands."
        )
        guidelines.append(
            "When summarizing your actions, output plain text directly; do not use "
            "bash to display what you did."
        )

    guidelines.append("Be concise in your responses")
    guidelines.append("Show file paths clearly when working with files")
    return guidelines


def _format_guidelines(guidelines: Sequence[str]) -> str:
    seen: set[str] = set()
    rendered: list[str] = []
    for guideline in guidelines:
        normalized = guideline.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        rendered.append(f"- {normalized}")
    return "\n".join(rendered)


def _format_context_files(context_files: Sequence[ContextFile]) -> str:
    if not context_files:
        return ""

    parts = [
        "# Project Context",
        "",
        "Project-specific instructions and guidelines:",
        "",
    ]
    for context_file in context_files:
        parts.append(f"# {context_file.path.name} instructions for {context_file.path.parent}")
        parts.append("")
        parts.append("<INSTRUCTIONS>")
        parts.append(context_file.content.rstrip())
        parts.append("</INSTRUCTIONS>")
        parts.append("")
    return "\n".join(parts).rstrip()


def format_skills_for_system_prompt(skills: Sequence[Any]) -> str:
    """Format skill metadata for model-visible discovery.

    The full skill body is intentionally not inlined here. The model sees the
    skill name, description, and location; direct slash skill invocation can
    include the full file content in the user turn.
    """

    if not skills:
        return ""

    lines = [
        "The following skills provide specialized instructions for specific tasks.",
        "Use a skill when the task matches its description.",
        "",
        "<available_skills>",
    ]
    for skill in skills:
        lines.extend(
            [
                "  <skill>",
                f"    <name>{_escape_xml(skill.name)}</name>",
                f"    <description>{_escape_xml(skill.description)}</description>",
                f"    <location>{_escape_xml(str(skill.path))}</location>",
                "  </skill>",
            ]
        )
    lines.append("</available_skills>")
    return "\n".join(lines)


def build_system_prompt(
    *,
    tools_by_name: Mapping[str, Tool],
    context_files: Sequence[ContextFile] | None = None,
    skills: Sequence[Any] | None = None,
    cwd: Path | None = None,
    permission_mode: PermissionMode = PermissionMode.YOLO,
    current_date: date | None = None,
) -> str:
    resolved_cwd = (cwd or Path.cwd()).resolve()
    today = current_date or date.today()
    loaded_context_files = (
        list(context_files) if context_files is not None else load_context_files(cwd=resolved_cwd)
    )

    sections = [
        DEFAULT_SYSTEM_PROMPT,
        f"Available tools:\n{_format_tools(tools_by_name)}",
        (
            "In addition to the tools above, you may have access to other custom "
            "tools depending on the project."
        ),
        VALIDATION_DISCIPLINE_PROMPT,
        (f"Guidelines:\n{_format_guidelines(_guidelines(tools_by_name, permission_mode))}"),
    ]

    context = _format_context_files(loaded_context_files)
    if context:
        sections.append(context)

    skills_prompt = format_skills_for_system_prompt(skills or [])
    if skills_prompt:
        sections.append(skills_prompt)

    sections.append(f"Current date: {today.isoformat()}")
    sections.append(f"Current working directory: {resolved_cwd}")
    return "\n\n".join(section for section in sections if section)


def _escape_xml(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )
