# `.claude/skills/` — index

Skills are markdown files with YAML frontmatter (`name` + `description`) that Claude Code reads on demand. When a task matches a skill's description, Claude invokes the skill and follows its content.

You can also read them as plain Markdown — they're useful reference for any agent or human contributor.

> See **AGENTS.md → "Capturing learnings"** for when to write a skill vs. a memory vs. an `AGENTS.md` rule vs. a hook.

## Skills in this repo

### Project knowledge (read first when picking up new work)

| Skill | Use when… |
|---|---|
| **`relann-repo-overview.md`** | First-time orientation — what `relann` is, repo layout, main components (parser, term_graph, era_operations, relnn, engine, session, tensor_term_compiler), and the compilation pipeline. |
| **`relann-conventions.md`** | About to edit any `relann/*.py` file — covers the juplit workflow, absolute-import rule for source modules, code style, debugging via `checkLogs`, row-first tensor layout, the no-patches policy, HGT shape invariants, and the inline-demo quirks. |
| **`relann-dsl-reference.md`** | Reading or writing RelaNN DSL — every construct (rule syntax, transforms, templates, operators, fit/predict, encode/decode brackets) with examples. |

### Tools and process

| Skill | Use when… |
|---|---|
| **`write-relnn-program.md`** | Authoring a new RelaNN program from a target architecture (PyTorch / paper formula → DSL rules). |
| **`juplit-programming.md`** | Working with `.py` ↔ `.ipynb` paired notebooks — cell delimiters, `if test():` blocks, `poe` task commands. |
| **`juplit-migrate-from-nbdev.md`** | Migrating an nbdev codebase to juplit (only relevant for outside repos — this one is already migrated). |
| **`planning-and-git-policy.md`** | Before starting any non-trivial change — asks clarifying questions first, never push without explicit permission, commit only when asked. |
| **`design-and-testing-doc-discipline.md`** | Whenever repo layout or test taxonomy changes — keeps `docs/design/repo-structure.md`, `docs/design/testing-strategy.md`, `TESTING.md`, and `tests/README.md` aligned. |

## Adding a new skill

1. Create a new `.claude/skills/<name>.md` with YAML frontmatter:
   ```
   ---
   name: <name>
   description: One-line description that tells Claude when to invoke this skill.
   ---
   ```
2. Body: the actual guidance — be specific, include code snippets / examples.
3. Add a row to the table above so humans can find it.
4. (Optional) Link from `CLAUDE.md` and `AGENTS.md` if it's a heavily-used skill.

## Skill format

```
---
name: skill-slug
description: When to use this skill (Claude reads this line to decide whether to invoke).
---

# Skill Title

Body of the skill. Markdown. Include:
- Concrete examples
- Specific file paths and line numbers when relevant
- Code snippets that can be copied verbatim
- Pitfalls and gotchas
```
