# Contributing

## Setup

```bash
git clone https://github.com/Alfredo-Sandoval/Illustrate.git
cd Illustrate
make env
pip install -e ".[gui]"
```

## QA

```bash
make lint        # ruff check .
make typecheck   # ty check
make test        # pytest
make qa          # lint + typecheck + test
make docs-check  # mkdocs build --strict
```

`make env` installs the tools required by these commands.

## Project layout

```
illustrate/            Core renderer, parser, IO, presets
illustrate_gui/        Desktop app (PySide6)
illustrate_web/api/    FastAPI routes + models
docs/                  MkDocs content
tests/                 Pytest suite
```

Keep rendering/domain logic in `illustrate/`. Keep UI-specific behavior in frontend packages.

## Conventions

- Keep public function signatures type-annotated.
- Avoid speculative fallbacks and broad exception suppression.
- Keep changes scoped: runtime logic, docs, and generated artifacts should not be mixed unless necessary.

## Submitting

1. Create a branch.
2. Run `make qa && make docs-check`.
3. Open a PR describing behavior changes and validation performed.
