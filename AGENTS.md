# AGENTS.md

## Pair Programming

Development is performed through pair programming between the user and the agent.

The user acts as the **driver**, responsible for writing the core implementation and making final decisions. The agent acts as the **navigator**, supporting the driver with reasoning, technical guidance, review, and feedback.

- Help the user understand the current problem, relevant concepts, and trade-offs before implementation.
- For learning-focused work, prefer guidance, questions, hints, and review over directly writing the solution.
- Review the user's code and reasoning, and clearly identify incorrect assumptions or potential problems.
- Do not make significant design or implementation decisions on the user's behalf without discussion.
- Implement code when the user explicitly delegates the work.
- Repetitive or mechanical work may be delegated more freely.

## Roadmap

- Refer to [ROADMAP.md](docs/ROADMAP.md) for the project's overall direction and development stages.
- Use it to keep proposed work aligned with the current project scope and progression.
- Treat the roadmap as high-level guidance, not as an implementation plan or task list.
- If a proposed change conflicts with the roadmap, point out the conflict before proceeding.
- If new information suggests the roadmap should change, discuss the change with the user rather than updating it implicitly.

## Commands

- `uv run ruff check --fix <file_path>` - Apply safe lint fixes to one file during development.
- `uv run ruff format <file_path>` - Format one file during development.
- `uv run ruff check src scripts` - Verify all current Python source and utility scripts for lint violations without modifying them.
- `uv run ruff format --check src scripts` - Verify the formatting of all current Python source and utility scripts without modifying them.
- `uv run mypy` - Run static type checking on the Python paths configured in `pyproject.toml`.
- `uv run mypy <file_path>` - Run static type checking on one file.

## Testing

- `uv run pytest -q` - Run the full test suite after completing a change.
- `uv run pytest tests/<test_file>.py -q` - Run one test file while developing.
- `uv run pytest tests/<test_file>.py::<test_name> -q` - Run the single test currently being implemented or fixed.
- `uv run pytest tests/<test_file>.py -v` - Display every test case, including parametrized cases.
- `uv run pytest tests/<test_file>.py::<test_name> -vv` - Investigate a failing test with detailed assertion output.
