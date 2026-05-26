# Skills

Wattle skills are reusable local capabilities stored as `SKILL.md` files. A skill is discovered at startup, listed in the system prompt by name and description, and loaded only when explicitly invoked.

## Layout

User skills:

```text
~/.wattle/skills/<skill-name>/SKILL.md
```

Project skills:

```text
.wattle/skills/<skill-name>/SKILL.md
```

Wattle searches from the current working directory up through parent directories, plus the user skill directory. More specific project skills override broader project or user skills with the same name.

## Skill file

```markdown
---
name: review
description: Review a code change for regressions and missing tests.
---

# Review Skill

Inspect the diff first. Prioritize concrete bugs, behavioral regressions,
and missing tests. Report findings with file and line references.
```

Frontmatter supports `name` and `description`. If no description is provided, Wattle uses the first heading or paragraph.

## Invocation

Invoke a skill with a slash command:

```text
/review inspect the current diff
```

Wattle expands the skill body into model-visible context and appends the task text. Built-in slash commands such as `/model`, `/help`, and `/compact` keep their normal behavior.

## Suggestions

The TUI can render slash-command suggestions for discovered skills. Skill names should be short, command-like, and stable.

## What skills are good for

Use skills for workflows that have repeatable judgment or process:

- code review standards
- release note style
- debugging playbooks
- framework-specific migration steps
- repository-specific deployment checks

Keep skills focused. A good skill teaches Wattle how to do one kind of work well.
