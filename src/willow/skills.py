"""Discovery and explicit invocation helpers for Willow skills.

Willow discovers skills from ``~/.willow`` and from ``.willow`` directories
walking from the current working directory up through its parents. Skills use
the directory layout ``.willow/skills/<skill-name>/SKILL.md``. Skill files may
include a simple YAML-like frontmatter block with ``name`` and ``description``
fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SKILL_FILE_NAME = "SKILL.md"


@dataclass(frozen=True)
class Skill:
    """Metadata for one discovered skill."""

    name: str
    description: str
    path: Path
    location: str

    def load_content(self) -> str:
        """Load the skill body from disk."""
        return self.path.read_text(encoding="utf-8")


class SkillNotFoundError(LookupError):
    """Raised when an explicit skill invocation cannot be resolved."""


def load_available_skills(cwd: str | Path) -> list[Skill]:
    """Discover user and ancestor project skills visible from ``cwd``.

    More specific project skills override broader project and user skills with
    the same name. Results are sorted by name for stable prompt formatting and
    TUI suggestions.
    """
    by_name: dict[str, Skill] = {}
    for location, root in _skill_roots(cwd):
        for path in _iter_skill_files(root):
            skill = _read_skill_metadata(path, location=location)
            if skill.name:
                by_name[skill.name.casefold()] = skill
    return sorted(by_name.values(), key=lambda skill: skill.name.casefold())


def format_skills_for_system_prompt(skills: list[Skill]) -> str:
    """Format skill metadata without inlining skill bodies."""
    if not skills:
        return "No Willow skills discovered."

    lines = [
        "Available Willow skills:",
        "Invoke a skill directly with /<skill_name> to load its body.",
    ]
    for skill in skills:
        description = f" - {skill.description}" if skill.description else ""
        lines.append(f"- {skill.name} ({skill.location}: {skill.path}){description}")
    return "\n".join(lines)


def resolve_skill(name: str, cwd: str | Path) -> Skill:
    """Resolve a skill by exact case-insensitive name."""
    wanted = name.strip().casefold()
    for skill in load_available_skills(cwd):
        if skill.name.casefold() == wanted:
            return skill
    raise SkillNotFoundError(f"Unknown skill: {name}")


def expand_skill_invocation(text: str, cwd: str | Path) -> str | None:
    """Expand ``/<skill_name> [task]`` into model-visible skill context.

    Returns ``None`` for non-skill input, including unknown slash commands, so
    callers can let normal command handling continue.
    """
    stripped = text.strip()
    if not stripped.startswith("/"):
        return None

    command, _separator, rest = stripped.partition(" ")
    skill_name = command.removeprefix("/")
    if not skill_name:
        return None

    try:
        skill = resolve_skill(skill_name, cwd)
    except SkillNotFoundError:
        return None

    content = skill.load_content().strip()
    task_text = rest.strip() or "(no task text provided)"
    return (
        f"Use the Willow skill {skill.name!r} from {skill.path}.\n\n"
        "<skill>\n"
        f"  <name>{_escape_xml(skill.name)}</name>\n"
        f"  <path>{_escape_xml(str(skill.path))}</path>\n"
        f"{content}\n"
        "</skill>\n\n"
        f"User task:\n{task_text}"
    )


def render_skill_suggestions(prefix: str, cwd: str | Path, *, limit: int = 8) -> str:
    """Render matching skill names for TUI suggestion rows."""
    stripped = prefix.strip()
    if not stripped.startswith("/"):
        return ""
    command_prefix = stripped.split(maxsplit=1)[0].removeprefix("/")
    query = command_prefix.casefold()
    rows: list[str] = []
    for skill in load_available_skills(cwd):
        if query and not skill.name.casefold().startswith(query):
            continue
        description = f"  {skill.description}" if skill.description else ""
        rows.append(f"/{skill.name}{description}")
        if len(rows) >= limit:
            break
    return "\n".join(rows)


def _iter_skill_files(root: Path) -> list[Path]:
    skills_root = root / "skills"
    if not skills_root.is_dir():
        return []

    paths: list[Path] = []
    for child in sorted(skills_root.iterdir(), key=lambda path: path.name.casefold()):
        if child.is_dir():
            skill_file = child / SKILL_FILE_NAME
            if skill_file.is_file():
                paths.append(skill_file)
    return paths


def _skill_roots(cwd: str | Path) -> list[tuple[str, Path]]:
    resolved_cwd = Path(cwd).expanduser().resolve()
    home = Path.home().expanduser().resolve()
    roots: list[tuple[str, Path]] = [("user", home / ".willow")]
    seen = {(home / ".willow").resolve()}

    ancestor_dirs = [resolved_cwd, *resolved_cwd.parents]
    if home in ancestor_dirs:
        ancestor_dirs = ancestor_dirs[: ancestor_dirs.index(home) + 1]

    for directory in reversed(ancestor_dirs):
        root = directory / ".willow"
        resolved_root = root.resolve()
        if resolved_root in seen:
            continue
        roots.append(("project", root))
        seen.add(resolved_root)
    return roots


def _read_skill_metadata(path: Path, *, location: str) -> Skill:
    content = path.read_text(encoding="utf-8")
    metadata, body = _split_frontmatter(content)
    default_name = path.parent.name
    name = metadata.get("name", default_name).strip()
    description = metadata.get("description", "").strip()
    if not description:
        description = _first_heading_or_paragraph(body)
    return Skill(name=name, description=description, path=path, location=location)


def _split_frontmatter(content: str) -> tuple[dict[str, str], str]:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, content

    metadata: dict[str, str] = {}
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return metadata, "\n".join(lines[index + 1 :])
        key, separator, value = line.partition(":")
        if separator and key.strip() in {"name", "description"}:
            metadata[key.strip()] = value.strip().strip("\"'")
    return {}, content


def _first_heading_or_paragraph(content: str) -> str:
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
        return stripped
    return ""


def _escape_xml(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )
