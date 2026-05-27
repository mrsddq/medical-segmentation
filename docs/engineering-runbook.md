# Engineering Runbook

## Repository Profile

- Repository: $repoName
- Classification: Python project
- Tracked files: 31
- Python files: 14
- JavaScript/TypeScript files: 0
- Notebooks: 0
- Terraform files: 0

## Setup

``bash
python -m pip install -r requirements.txt
``

## Verification

``bash
python -m unittest discover -s tests
python -m compileall -q .
``

## Release Hygiene

- Keep generated outputs, caches, local datasets, virtual environments, and dependency folders out of git.
- Prefer deterministic commands over manual notebook or console-only steps.
- Document required secrets and environment variables instead of committing them.
- Keep Dockerfiles, CI workflows, and tests aligned with the actual project stack.
- Treat learning or reference material honestly as reference material; do not present it as production service code unless it has service-grade tests, deployment, and operations docs.

## Maintenance Checklist

- Review dependencies quarterly.
- Run tests before every push.
- Confirm git status --short is clean before packaging.
- Include .git only when an external submission explicitly requires repository history.
