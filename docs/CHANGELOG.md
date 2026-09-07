# Implementation audit changelog

## Second audit and integration pass

- Added the article's Episode–ContextObject fusion graph, typed edge construction cascade, graph-aware GC, and validated retrieval tuning.
- Added the six-step planner prompt, confidence-based rule/encoder/local/planner routing, and explicit abstention.
- Added nonblocking `start_gc` / `poll_gc`, model-triggered `compress_context`, availability-based mode switching, and context-manager cleanup.
- Fixed user span permissions, log tail preservation, aged failed-input purging, known instruction/write protection, and media boundaries.
- Preserved live dependencies during retrieval; revived previously hidden evidence when a later task references it; rejected requests if the session changes during retrieval.
- Added per-zone cache TTL simulation, planning-time expected value, and append-only compaction request construction.
- Added deterministic Ollama tool-message conversion: argument strings become JSON objects and tool results use `tool_name`, following the [official tool-calling format](https://docs.ollama.com/capabilities/tool-calling).
- Added source-cited turn JSON with an exact constraint spine, a conservative information gate, atomic memory-source merging, optional semantic recall, and symbol search.
- Added cross-session symbolic pattern consolidation and benchmark/golden-trace/skill candidates. They remain auditable candidates, not trained models or installed skills.
- Removed repeated BM25 document-frequency scans; reused file-tracker snapshot lookups; expired old cache entries; replaced recursive alias recovery with a validated iterative traversal.
- Added 44-item machine-checkable traceability, a second end-to-end methodology lab, extension protocols, `python -m context_lab`, English README, and a Chinese principles-to-code guide.

The changes do not claim official paper reproduction, live neural model validation, provider billing accuracy, production host integration, or binary multimodal recovery.


## Reliability and explanation refinement

- Pack complete evidence with source IDs against the serialized request, body budget and optional output reserve; expose selected and omitted candidates.
- Resume persisted deferred plans after restart, without rerunning the planner; compare both source revision and proposal identity at commit; discard only the stale proposal.
- Validate then atomically write each memory extraction window; support explicit correction targets; reject correction cycles and inconsistent raw-source records.
- Batch query/document encoding across expanded memory searches; do not assign RRF evidence scores to zero-match results.
- Add an offline reliability walkthrough, targeted regression tests, and substantive additions to the main Chinese explanation, including chapter 19.
