---
name: python-via-uv
description: Enforce Python execution through `uv` instead of direct interpreter calls. Use when Codex needs to run Python scripts, modules, one-liners, tools, test runners, or package commands in a workspace and should avoid invoking `python` or `python3` directly.
---

# Python Via Uv

Use `uv` for every Python command.

Do not run `python`.
Do not run `python3`.
Do not suggest `python` or `python3` in instructions unless the user explicitly requires them and the constraint must be called out as a conflict.

## Execution Rules

When sandboxed, set `UV_CACHE_DIR` to a temporary directory the agent can write to before running `uv` commands.

Prefer these patterns:

- Run a script: `UV_CACHE_DIR=/tmp/uv-cache uv run path/to/script.py`
- Run a module: `UV_CACHE_DIR=/tmp/uv-cache uv run -m package.module`
- Run a one-liner: `UV_CACHE_DIR=/tmp/uv-cache uv run python -c "print('hello')"`
- Run a tool exposed by dependencies: `UV_CACHE_DIR=/tmp/uv-cache uv run tool-name`
- Add a dependency for an ad hoc command: `UV_CACHE_DIR=/tmp/uv-cache uv run --with <package> python -c "..."`

## Notes

Using `python` inside `uv run ...` is acceptable because `uv` is still the entrypoint controlling interpreter selection and environment setup.

If the workspace already defines a project-specific temporary cache directory, prefer that over `/tmp/uv-cache`.

If a command example or existing documentation uses `python` or `python3` directly, translate it to the closest `uv` form before executing it.

Use `uv help run` to get full help about the command `uv run` when needed.
