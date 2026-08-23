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

## Design Decisions

- Discuss one meaningful design decision at a time before implementation.
- Do not merely accept the user's proposal. Evaluate it using technical evidence, explain disagreements, and make a reasoned recommendation while leaving the final decision to the user.
- For each meaningful decision, identify the current problem and constraints, compare viable alternatives and trade-offs, state the chosen approach and rejected alternatives, and record when the decision should be revisited.
- Treat established libraries and frameworks as references, not authorities. Reuse a name or structure only when its responsibility and semantics match this project; otherwise adapt it to the local problem.
- When a decision depends on an external API, protocol, framework, or library, verify its current official documentation instead of relying on memory or analogy.
- Balance speculative generalization against short-sighted implementation. Build only what the current scope requires, but identify known future requirements as revisit triggers when they may invalidate the current choice.

## Acceptance Criteria

- Distinguish structural contracts, such as module boundaries and dependency direction, from observable behavior, such as outputs, state changes, repetition, and termination.
- Write behavioral acceptance criteria as precise, observable outcomes, including relevant boundary conditions. Avoid wording whose meaning depends on an unstated interpretation.
- When discussion changes an acceptance criterion, scope boundary, or design decision, keep the corresponding issue and pull request description synchronized.

## Testing

### Principles

- Do not force a one-to-one mapping between checklist items and tests. Before adding a test, identify the requirement or risk it covers, the specific failure or regression it catches, and why existing tests would not catch that failure.
- Do not add or retain a test that has no distinct failure-detection value.
- Prefer assertions about outcomes and state over implementation interactions. Assert call order or another interaction only when that interaction is itself part of the contract.
- Use RED-GREEN-REFACTOR for new behavior and bug fixes. When a test is added after the behavior already works, describe it accurately as acceptance or regression coverage rather than claiming a TDD cycle.

### Naming

- Test functions start with `test_` and use English ASCII `snake_case`.
- Name tests after the behavior they guarantee, not an internal method or implementation step.
- After `test_`, express the stable subject or capability, the expected observable behavior or outcome, and then any condition needed to distinguish that behavior.
- Express expected behavior with an active present-tense verb such as `returns`, `raises`, `rejects`, `preserves`, or `stops`. Choose a verb that matches the actual observable contract.
- Use `when` for general conditions. Use a more precise connector such as `before`, `after`, `without`, or `from` when it expresses the relationship more accurately. Omit the condition when it is unnecessary.
- Include only conditions that distinguish the expected result. Include a concrete ID or value only when that value is itself part of the contract or a boundary condition.
- Mention calls, ordering, or data transfer only when that interaction is an explicit protocol or acceptance contract.
- Treat `and` as a signal to check whether the test covers multiple independent behaviors. Keep it only when both clauses form one indivisible workflow contract.
- For parametrized tests, let the function name describe the behavior shared by every case. Add an explicit domain-oriented parameter ID only when pytest's generated ID does not identify the case clearly.
- Do not impose a fixed character limit. Preserve the expected behavior and important condition, and investigate multiple behaviors or irrelevant setup details before shortening a long name with abbreviations.
- Use canonical project and domain terminology consistently. Do not introduce a synonym for an existing concept, and use abbreviations only when they are established in the project or broadly understood.
- Apply these rules to new or meaningfully modified tests. Do not bulk-rename existing tests solely for stylistic consistency.

Examples:

- `test_agent_returns_final_text_without_tool_calls`
- `test_get_order_returns_not_found_when_order_id_is_unknown`
- `test_agent_stops_before_executing_tool_from_fifth_model_call`

Avoid:

- `test_get_order_works`
- `test_should_return_error`
- `test_customer_001_order_999`

These rules are based on [pytest's test discovery conventions](https://docs.pytest.org/en/stable/explanation/goodpractices.html#conventions-for-python-test-discovery), [PEP 8](https://peps.python.org/pep-0008/#function-and-variable-names), the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#s3.16-naming), [*Software Engineering at Google*](https://abseil.io/resources/swe-book/html/ch12.html#name-tests-after-the-behavior-being-tested), and [Microsoft unit testing best practices](https://learn.microsoft.com/en-us/dotnet/core/testing/unit-testing-best-practices#follow-test-naming-standards).

## Commands

### Linting and Formatting

- `uv run ruff check --fix <file_path>` - Apply safe lint fixes to one file during development.
- `uv run ruff format <file_path>` - Format one file during development.
- `uv run ruff check src scripts` - Verify all current Python source and utility scripts for lint violations without modifying them.
- `uv run ruff format --check src scripts` - Verify the formatting of all current Python source and utility scripts without modifying them.

### Type Checking

- `uv run mypy` - Run static type checking on the Python paths configured in `pyproject.toml`.
- `uv run mypy <file_path>` - Run static type checking on one file.

### Testing

- `uv run pytest -q` - Run the full test suite after completing a change.
- `uv run pytest tests/<test_file>.py -q` - Run one test file while developing.
- `uv run pytest tests/<test_file>.py::<test_name> -q` - Run the single test currently being implemented or fixed.
- `uv run pytest tests/<test_file>.py -v` - Display every test case, including parametrized cases.
- `uv run pytest tests/<test_file>.py::<test_name> -vv` - Investigate a failing test with detailed assertion output.
