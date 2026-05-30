---
name: design-and-testing-doc-discipline
description: Use before changing the repo's test layout, test profiles/buckets, or design docs. Encodes the rule that if repo layout or test taxonomy changes, the matching design doc (docs/design/repo-structure.md, docs/design/testing-strategy.md, TESTING.md, tests/README.md) must be updated in the same change.
---

# Design and Testing Doc Discipline

- If repository layout ownership changes, update `docs/design/repo-structure.md`.
- If test taxonomy, buckets, profiles, or run commands change, update:
  - `docs/design/testing-strategy.md`
  - `TESTING.md`
  - `tests/README.md`
- Execution plans (future plan files) are execution-only and must include:
  - `Design refs: docs/design/repo-structure.md, docs/design/testing-strategy.md`
- Do not duplicate architecture policy in plans; link to the design docs instead.
- Keep test execution simple: prefer `uv run poe smoke` / `quick` / `test` or `scripts/run_tests.py` profiles (`smoke`, `hgt`, `dhn`, `full`).
