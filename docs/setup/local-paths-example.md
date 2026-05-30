---
description: "Local Python/env paths (copy to local.mdc and fill in)"
alwaysApply: false
---

# Local environment (template)

Copy this file to `local.mdc` and set the paths for your machine.

**Use this Python executable for all project commands** (tests, scripts, etc.):

```
<path-to-conda-env>/python.exe
```

**Use this script for nbdev_prepare** (export notebooks to lib, run tests, etc.):

```
<path-to-conda-env>/Scripts/nbdev_prepare
```
(On Windows the file is `nbdev_prepare.exe`; on WSL use `.../Scripts/nbdev_prepare.exe`.)

- Use the **parent** conda environment for this repo.
- When running Python (e.g. pytest, scripts), use the Python path above.
- When running nbdev_prepare, use the Scripts path above so the correct environment is used.
