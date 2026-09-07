# Final validation record / 最终验证记录

Date: 2026-09-07. This records checks actually executed in the local workspace, not a claim about upstream paper results.

| Check | Result | What it establishes |
|---|---|---|
| `uv run pytest -q` | 55 passed | Behavioral regression and integration tests |
| `uv run ruff check src tests examples main.py` | Passed | Configured lint rules |
| `uv run context-lab coverage --check` | 44 entries; no missing references | Implementation/test symbols exist; not line coverage |
| `uv run context-lab demo` | Passed | Core offline experiment with persisted artifacts |
| `uv run context-lab methodology` | Passed | Combined architecture experiment and mode switching |
| README CLI smoke checks | Passed | Ingest, plan/commit/recover, memory query, turn summary, compaction request, code-map search/read |
| `uv run python examples/reliability_walkthrough.py` | Passed | Deferred restart, full-request packing, targeted corrections |
| Local Markdown targets | Passed | README, docs and main guide local links/explicit anchors resolve in this workspace |
| `uv build` | sdist and wheel built | Package build configuration works |
| Isolated wheel import | Passed | Runtime imports from the built wheel, not the editable checkout |
| Source archive inspection | Passed | Reference PDFs/HTML, scratch data and run state are excluded |

## What the tests cover

Storage immutability, text archive integrity, alias recovery, session scope, instruction and side-effect protection, exact sparse evidence, user fold-only permissions, log tail preservation, aged error-input purging, tool protocol integrity, stale plans, pending-plan restart and identity checks, nonblocking scheduling, retrieval races, dependency revival, serialized evidence packing and output reserve, budget checks, graph construction/tuning/liveness, intent-sensitive paths, structured summary provenance, memory source merging under concurrent writes, targeted corrections, cycle rejection and atomic memory windows, temporal anchoring, cross-session archive scope, heartbeat paging, code-map versions/search, cache TTL/prices/forecasts, Ollama transport/message conversion, optional adapter contracts, clean-trace candidates and complete offline reports.

## What was not validated

No neural model weights were downloaded or evaluated. Ollama HTTP tests use a local test server; encoder, embedding and LLMLingua adapter tests use controlled substitutes to verify contracts. Model quality, semantic faithfulness, benchmark scores, real provider cache hits and billing, learned classification calibration, binary multimodal recovery, fine-tuning/RL, and production host integration remain outside this validation.

## Reproduce

```bash
uv sync --group dev
uv run ruff check src tests examples main.py
uv run pytest -q
uv run context-lab coverage --check
uv run context-lab demo
uv run context-lab methodology
uv run python examples/reliability_walkthrough.py
uv build
```

Each report run creates its own directory. User reference materials remain separate from package distributions. The source manifest is [coverage.json](coverage.json); method-by-method limits are in [the audit](二次实现审计.md).
