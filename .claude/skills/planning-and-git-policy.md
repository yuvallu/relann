---
name: planning-and-git-policy
description: Use before starting any non-trivial feature or change in this repo, and before any git commit or push. Lays out the planning discipline (ask clarifying questions first, get confirmation, only then code) and the git policy (never push without explicit user permission; commit only when asked).
---

# Planning and Git Policy

## Ask questions before building

When planning a new feature or significant change:
1. Read the relevant code and design docs.
2. **Ask the user clarifying questions** before starting implementation. Cover scope, edge cases, naming, trade-offs, and anything ambiguous.
3. Only begin coding after the user confirms the plan.

Skipping questions leads to rework. When in doubt, ask.

## Git commit / push policy

- **Never push** (`git push`) unless the user explicitly says to push.
- After committing, let the user review locally first.
- It is fine to `git commit` when asked, but do not follow it with `git push` automatically.
