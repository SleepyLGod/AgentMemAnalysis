# Solution Landscape Analysis

For each problem identified in `problems.md`, this document analyzes whether **Flink**, **classical DB techniques**, **LOTUS**, and **Continuous Prompts (CP)** can help, and where they fall short.

## Background: Key Capabilities

| System/Technique | Core Strengths | Key Limitations |
|---|---|---|
| **Apache Flink** | Tumbling/sliding/session windows; stateful processing w/ exactly-once; incremental aggregation (`AggregateFunction`); watermark + late data; keyed streams for per-key parallelism; async I/O; `ML_PREDICT` for external LLM calls | No semantic-level optimization; treats LLM as opaque external service; no cost/accuracy modeling for LLM calls |
| **Classical DB** | CBO (cost-based optimizer); materialized views & IVM (incremental view maintenance); predicate pushdown; operator fusion; query sharing / CSE (common sub-expression elimination); join ordering; adaptive query processing | Assumes structured data; no native "semantic predicate" support; optimization rules need new cost models for LLM operators |
| **LOTUS** | Declarative semantic operators (`sem_map`, `sem_filter`, `sem_join`, `sem_topk`, `sem_agg`); **model cascade** (proxy → gold LLM); batched inference (multi-row → one LLM call); accuracy guarantees via statistical bounds | **Batch-only** — no streaming/incremental support; optimizer is **intra-operator only** (no cross-operator reordering/fusion); no graph-aware semantics; no stateful prompt management |
| **Continuous Prompts (CP)** | Extends semantic operators to **streaming** (first framework bridging LLM + continuous query); **semantic windows** (LLM-driven boundary detection w/ continuity scores); **semantic group-by** (incremental clustering w/ LLM refinement); **continuous RAG** (persistent retrieval over evolving streams); **tuple batching** (multi-tuple → one LLM call); **operator fusion** (multi-operator → one prompt); **dynamic planning** via MOBO for accuracy-throughput Pareto frontier | Focuses on **pipeline-level** streaming (filter/map/topk chains); no graph operations; limited to operators on flat tuples; operator fusion trades accuracy; no explicit CBO with cost models for LLM token pricing; still early-stage (evaluated on classification/news monitoring tasks) |

---

## 1. EverMemOS

### 1.1 Tumbling Window Boundary Detection Latency

| Approach | Analysis |
|----------|----------|
| **Flink** | ✅ **Strong fit.** Session windows + watermarks handle this natively. Can replace LLM-gated boundary detection with gap-timeout-based session segmentation. Flink's `ProcessFunction` + keyed state enables incremental boundary detection (rolling state, not full history replay). |
| **Classical DB** | **IVM**: Treat boundary detection as an incremental view — process only the delta (new messages), not the full accumulated history. **Query sharing**: Batch concurrent conversations' boundary requests. |
| **LOTUS** | ❌ Batch-only. No streaming window semantics. |
| **CP** | ✅ **Directly applicable.** CP's **semantic window (ωs)** is precisely this problem — LLM-driven boundary detection over streams. CP offers three strategies: (a) pairwise continuity scoring (compare consecutive msgs); (b) summary-based windowing (maintain evolving summary, check new msg against it); (c) embedding-based windowing (clustering). The **summary-based** variant avoids sending full history by maintaining a rolling summary, directly addressing the prompt-bloat issue. CP's dynamic planning can also adaptively switch between LLM-based and embedding-based windowing based on throughput needs. |

### 1.2 Unbounded Profile State Growth

| Approach | Analysis |
|----------|----------|
| **Flink** | Partial. Flink manages large state well (RocksDB backend), but the bottleneck is "entire profile in LLM prompt", not storage. |
| **Classical DB** | ✅ **Materialized view + IVM**: Maintain profile as a materialized view; only do incremental updates (diff-based `sem_agg`). **Hierarchical aggregation**: Maintain per-topic sub-profiles; update only affected sub-topics. |
| **LOTUS** | ❌ LOTUS's `sem_agg` is supported but has no incremental aggregation mode. |
| **CP** | **Partial.** CP's `sem_agg` supports an incremental mode that "maintains evolving state as new tuples arrive," which could be adapted for incremental profile updates. However, CP doesn't explicitly address the "unbounded state object being passed as LLM context" problem — incremental aggregation still requires the accumulated summary in context. CP's summary-based semantic window shows a pattern (rolling summary), but doesn't solve the core token growth issue for aggregation state. |

### 1.3 Multi-Perspective Extraction Redundancy

| Approach | Analysis |
|----------|----------|
| **Flink** | Indirect — operator chaining reduces serialization overhead, but doesn't merge LLM calls. |
| **Classical DB** | ✅ **Operator Fusion / Multi-Output Query**: Three `sem_map/sem_extract` calls on the same input → fuse into one LLM call producing (episode, foresight, event_log). Classic scan-sharing problem. |
| **LOTUS** | Limited. LOTUS batched inference targets multi-row, not multi-output fusion on the same row. |
| **CP** | ✅ **Directly applicable.** CP's **operator fusion** combines adjacent operators into a single LLM prompt with a fused schema. Three parallel extractions (Episode, Foresight, EventLog) on the same text can be fused into one `map → map → map` chain, which CP showed yields ~1.3× speedup for map→map fusion. CP also quantifies the accuracy tradeoff of fusion: "fusion is safe for map→map pairs with low sensitivity." This is strong evidence that EverMemOS's 3-perspective extraction is fusible with controlled accuracy loss. |

### 1.4 Spiky Latency from Conditional Consolidation

| Approach | Analysis |
|----------|----------|
| **Flink** | ✅ **Incremental aggregation** (`AggregateFunction`): naturally converts batch-triggered consolidation into continuous incremental updates. |
| **Classical DB** | **Continuous aggregation / streaming materialized view**: Update profile on every message incrementally, eliminating the threshold-triggered spike. |
| **LOTUS** | ❌ Batch-only. No incremental aggregation. |
| **CP** | ✅ **Applicable.** CP's incremental `sem_agg` mode maintains evolving state and updates it per-tuple, which conceptually eliminates the batch-trigger pattern. However, CP doesn't explicitly discuss conditional triggering vs. continuous aggregation. The key benefit is its streaming-native design: aggregation state evolves continuously rather than accumulating until a threshold. |

### 1.5 Strict Temporal Dependencies

| Approach | Analysis |
|----------|----------|
| **Flink** | Partial. Per-key (per-conversation) parallelism is native. Within a single conversation, msg ordering is a business constraint Flink can't bypass. Session windows can reduce dependence on LLM-based boundary detection. |
| **Classical DB** | **Speculative/optimistic execution**: Assume no boundary, pre-process N+1, rollback if wrong. **Pipeline parallelism**: Overlap boundary detection with downstream extraction. |
| **LOTUS** | ❌ No streaming. |
| **CP** | **Partial.** CP's semantic windowing still requires per-tuple evaluation for boundary decisions, preserving the sequential dependency within a window. However, CP's **embedding-based windowing** variant sidesteps this by using cheap vector similarity (no LLM call per msg), enabling much higher throughput at some accuracy cost. CP's dynamic planner can switch to embedding-based windowing during high-throughput bursts. |

---

## 2. Mem0

### 2.1 Deeply Chained Serial Semantic Operations

| Approach | Analysis |
|----------|----------|
| **Flink** | Indirect. Async I/O helps parallelize DB lookups, but the serial chain is caused by data dependencies (entity extraction → relation extraction → dedup). |
| **Classical DB** | ✅ **CBO + operator reordering**: A semantic CBO could reorder operators (e.g., push cheap filters before expensive extraction), do **operator fusion** (fuse entity + relation extraction into one call), and **predicate pushdown** (inline dedup into extraction). |
| **LOTUS** | ✅ Provides the correct **declarative abstraction** (semantic query plan), making optimization possible. But LOTUS's current optimizer only does **intra-operator** cascade, **not cross-operator reordering/fusion**. |
| **CP** | ✅ **Operator fusion directly applicable.** CP explicitly studies fusing adjacent operators (e.g., map→map, map→filter) into one LLM call. The entity extraction → relation extraction chain is a `map→map` pattern, which CP showed is safely fusible with ~1.3× speedup and minimal accuracy loss. CP's **dynamic planning** could also learn the optimal fusion configuration for this specific chain. However, CP doesn't do full CBO-style **reordering** — it only fuses adjacent operators in the existing order. |

### 2.2 Long-Tail Latency in 1-to-N Mapping

| Approach | Analysis |
|----------|----------|
| **Flink** | ✅ `flatMap` + async I/O naturally handles 1-to-N fan-out with parallelism and backpressure. |
| **Classical DB** | **Lateral join / UNNEST** + adaptive parallelism based on fan-out cardinality. |
| **LOTUS** | Limited. Batched inference helps (multi-row → one call), but no adaptive handling of long-tail distribution. |
| **CP** | **Partial.** CP's tuple batching can batch the fan-out items into fewer LLM calls, reducing per-item overhead. But CP doesn't explicitly model adaptive batching based on fan-out cardinality — batch size is a global parameter, not per-tuple adaptive. |

### 2.3 Top-1 Retrieval Risk

| Approach | Analysis |
|----------|----------|
| **Flink** | ❌ Not directly relevant. |
| **Classical DB** | **Adaptive top-K**: Dynamically increase k based on confidence score of top-1 result. |
| **LOTUS** | ✅ **Model cascade** for `sem_topk` / `sem_sim_join`: proxy-based filtering naturally retrieves more candidates than hard top-1, then LLM verifies. Statistically guaranteed recall. |
| **CP** | **Indirect.** CP's continuous RAG with sub-prompting (SP-LLM) could provide more precise retrieval than a single unified query. But CP doesn't explicitly address the k-selection problem or provide cascade-style retrieval guarantees. CP's embedding-based variants (SP-Emb, UP-Emb) offer fast alternative retrieval but with potential accuracy loss. |

### 2.4 2-Stage Extraction Context Redundancy

| Approach | Analysis |
|----------|----------|
| **Flink** | ❌ Not directly relevant. |
| **Classical DB** | ✅ **Operator fusion / scan sharing**: Fuse entity + relation extraction into one LLM call. **CSE**: Share the raw-message prefix via KV cache. |
| **LOTUS** | Indirect. `sem_map` could potentially extract both in one call, but LOTUS doesn't discuss multi-stage fusion. |
| **CP** | ✅ **Directly applicable.** This is a textbook `map→map` fusion case in CP's framework. CP showed map→map fusion yields speedup with minimal accuracy loss. The fused prompt would be: "Extract entities AND their relations from this text" with a combined JSON schema. CP's accuracy profiling can quantify the tradeoff. |

---

## 3. Zep / Graphiti

### 3.1 Join Cost (sem_join Cartesian Product)

| Approach | Analysis |
|----------|----------|
| **Flink** | ❌ Not directly relevant. |
| **Classical DB** | ✅ **Semi-join reduction / bloom filter**: Pre-filter candidate pairs with cheap structural predicates before expensive LLM sem_join. **Join ordering**: Process cheapest joins first. |
| **LOTUS** | ✅ **Core strength.** LOTUS's `sem_join` with model cascade (embedding proxy → LLM gold) directly reduces LLM calls by up to 1000×. Most directly applicable optimization. |
| **CP** | **Partial.** CP doesn't explicitly define a streaming `sem_join` operator. Its continuous RAG could serve as a retrieval mechanism before join, but it lacks the cascade optimization that LOTUS provides for join specifically. CP's tuple batching could batch multiple join verifications into one LLM call, reducing per-pair overhead. |

### 3.2 Unconditional Summary Generation

| Approach | Analysis |
|----------|----------|
| **Flink** | ❌ Not directly relevant. |
| **Classical DB** | **Lazy/conditional materialized view**: Only refresh summary when new information is detected (CDC for novelty). |
| **LOTUS** | Indirect. Could add `sem_filter("Does this msg contain new info about entity X?")` before summary update, leveraging LOTUS's cascade optimization. |
| **CP** | ✅ **Applicable via fusion.** A `filter→map` fusion (filter = "is new info?", map = "update summary") can be fused into one CP prompt. CP showed filter→map fusion yields ~1.08× speedup. If the filter rejects most entities (low selectivity = most entities don't need update), CP's analysis shows unfused is better (avoid wasting LLM call on map for filtered items). This aligns well: the filter skips most entities, saving the expensive map. |

### 3.3 Static Communities

| Approach | Analysis |
|----------|----------|
| **Flink** | Conceptual. Flink's stateful processing enables incremental community maintenance on entity/edge changes — analogous to incremental label propagation. |
| **Classical DB** | ✅ **IVM**: Community as materialized view over entity graph; incremental refresh on delta changes. **Online clustering**: Replace batch label propagation with incremental algorithms. |
| **LOTUS** | ❌ No graph operations or community detection. |
| **CP** | **Partial.** CP's **semantic group-by (μs)** is conceptually similar — it does incremental clustering with LLM refinement, creating/merging/splitting groups dynamically as new data arrives. If entities are modeled as stream tuples, CP's group-by could serve as an online community assignment mechanism. However, CP's group-by operates on flat tuples, not graph topology — it wouldn't use edge connectivity, just entity text similarity. |

### 3.4 Iterative Pairwise Reduction

| Approach | Analysis |
|----------|----------|
| **Flink** | Conceptual. Flink's `reduce()` / tree aggregation provides the execution pattern, but the bottleneck is LLM latency per round. |
| **Classical DB** | ✅ **Single-pass `sem_agg`** to replace multi-round pairwise reduction. If too many summaries for one context window, use **two-level MapReduce aggregation**. |
| **LOTUS** | ✅ LOTUS's `sem_agg` is a single-pass semantic aggregation (subject to context window limits). |
| **CP** | ✅ **Applicable.** CP's incremental `sem_agg` with evolving state naturally replaces pairwise reduction — each new entity summary is incrementally folded into the community summary, one at a time, maintaining a rolling aggregate. This is strictly better than O(log M) rounds of pairwise reduction. CP also showed that agg fusion can be risky (accuracy-sensitive), so empirical validation is needed. |

### 3.5 N-to-1 Update Conflicts (Race Condition)

| Approach | Analysis |
|----------|----------|
| **Flink** | ✅ Keyed stream by `community_uuid` naturally serializes all updates to the same community on one operator instance. `AggregateFunction` merges multiple entity summaries incrementally. |
| **Classical DB** | ✅ **GROUP BY + sem_agg**: Collect all entity summaries for same community, then one-shot aggregate. **Serializable isolation** if per-entity update is kept. |
| **LOTUS** | ✅ `GROUP BY` + `sem_agg` directly supported. |
| **CP** | ✅ **Directly applicable.** CP's **semantic group-by** + `sem_agg` achieves the same: group entities by community, then aggregate. CP's group-by is streaming-native, so it handles concurrent arrivals naturally. The incremental aggregation mode ensures no race — updates are processed sequentially per group. |

### 3.6 Context Fragmentation in Batch Extraction

| Approach | Analysis |
|----------|----------|
| **Flink** | ❌ Not directly relevant — this is a context window limitation problem. |
| **Classical DB** | **Affinity-based partitioning / graph-aware chunking**: Partition nodes by topological proximity (co-occurrence, edge density) instead of arbitrary chunks. |
| **LOTUS** | ❌ Not applicable. |
| **CP** | ❌ CP doesn't address intelligent context-window partitioning. CP's tuple batching packs multiple tuples but doesn't solve the intra-prompt fragmentation of logically related entities. |

### 3.7 Edge Resolution Search Amplification (3N Retrievals)

| Approach | Analysis |
|----------|----------|
| **Flink** | ❌ Not directly relevant. |
| **Classical DB** | ✅ **Query sharing / multi-query optimization**: The three retrieval passes (structural lookup, filtered hybrid search, unfiltered hybrid search) share underlying BM25+vector computation — do **one hybrid search**, then apply two different post-filters in the app layer. **Batched query**: Batch N edges' retrievals into fewer DB round-trips. |
| **LOTUS** | Indirect. LOTUS's `sem_join` cascade could replace the search+LLM-verify pattern, but the underlying N DB lookups remain. |
| **CP** | **Partial.** CP's continuous RAG could maintain a persistent retrieval index that serves all edge queries incrementally, but CP doesn't optimize multi-query retrieval sharing. CP's tuple batching could batch N edge verification prompts into fewer LLM calls, reducing the N LLM calls to N/T calls. |

### 3.8 Fixed Previous Episode Window

| Approach | Analysis |
|----------|----------|
| **Flink** | ✅ Session/sliding windows provide adaptive context window sizing. |
| **Classical DB** | **Adaptive windowing**: Dynamically size the lookback based on entity density or topic similarity (index-based range scan vs. fixed limit). |
| **LOTUS** | ❌ No window strategy. |
| **CP** | ✅ **Directly applicable.** CP's **semantic window** dynamically adjusts boundaries based on content (topic drift, entity transitions), which is exactly the adaptive episode selection needed here. Instead of fixed "last 10 episodes," use CP's summary-based semantic window to select episodes that are semantically relevant to the current extraction context. |

---

## 4. Cross-Cutting Problems

### 4.1.1 LLM Cost Explosion in Top-K Retrieval

| Approach | Analysis |
|----------|----------|
| **Flink** | ❌ |
| **Classical DB** | ✅ **Adaptive top-K + cost-based K selection**: Dynamic k based on task importance and candidate quality distribution. |
| **LOTUS** | ✅ **Core strength.** `sem_topk` / `sem_join` with model cascade (embedding proxy → LLM gold) directly reduces LLM calls, with accuracy guarantees. |
| **CP** | **Partial.** CP has `sem_topk` with incremental mode, but doesn't implement LOTUS-style model cascades. CP could use embedding-based variants as cheap proxies (similar to cascade), but this is left to the dynamic planner rather than being a built-in algorithm. |

### 4.1.2 Hard-Coded / Cost-Unaware Workflows

| Approach | Analysis |
|----------|----------|
| **Flink** | ❌ |
| **Classical DB** | ✅ **This IS the CBO problem.** Need a semantic CBO with LLM cost models (tokens × $/token, latency, accuracy) for operator reordering, fusion, cascade selection. |
| **LOTUS** | ✅ Provides the declarative abstraction; optimizer does intra-operator cascade. But **no inter-operator** reordering/fusion/global CBO. |
| **CP** | ✅ **Strongest fit.** CP's **dynamic planning framework** with MOBO is the closest to a semantic CBO: it generates candidate plans (varying batch sizes, operator variants, fusion options), learns per-operator cost-accuracy models from telemetry, and selects Pareto-optimal configurations at runtime. This is more advanced than LOTUS's static intra-operator cascade — CP does **inter-operator** fusion decisions and adaptive plan reconfiguration. However, CP's optimizer is primarily throughput-accuracy focused, not token-cost focused. |

### 4.1.3 State-Dependent Prompt Bloat

| Approach | Analysis |
|----------|----------|
| **Flink** | Indirect. Flink's state management can maintain delta, but LLM prompts need full context semantically. |
| **Classical DB** | ✅ **IVM / delta propagation**: Pass only delta to LLM ("here's the old profile and new evidence, update"). **KV-cache-aware prompt design**: Maximize prefix stability for LLM serving cache hits. |
| **LOTUS** | ❌ Stateless per-query. |
| **CP** | **Partial.** CP's design uses rolling summaries (in semantic windows and incremental aggregation) which partially mitigates full-state re-submission. The summary-based semantic window, for example, maintains evolving summary state rather than raw history. But CP doesn't explicitly optimize for prefix-cache utilization or delta-based prompt construction. |

### 4.1.4 Missing Global Context in Resolution

| Approach | Analysis |
|----------|----------|
| **Flink** | Partial. Can do global dedup stage before per-fact resolution. |
| **Classical DB** | ✅ **Self-join elimination / CSE**: Pre-deduplicate facts within the same batch before issuing per-fact retrieval. Share LLM results when multiple facts retrieve the same candidates. |
| **LOTUS** | Limited. `sem_join` has dedup, but no cross-query CSE. |
| **CP** | **Partial.** CP's tuple batching groups multiple facts into one LLM call, which implicitly gives the LLM global context across the batch. If multiple facts reference the same entity, the batched prompt would contain all of them, allowing the LLM to deduplicate internally. However, this is an implicit benefit, not an explicit optimization. |

### 4.1.5 Unbounded Candidate Set / Missing Pushdown

| Approach | Analysis |
|----------|----------|
| **Flink** | ❌ |
| **Classical DB** | ✅ **Semantic predicate pushdown** (ideal but doesn't exist yet). **Bloom filter / metadata pre-filter** at storage layer to reduce candidate set before app-layer LLM evaluation. |
| **LOTUS** | ✅ **Semantic index**: LOTUS maintains semantic-aware indices for approximate pushdown. |
| **CP** | **Partial.** CP's continuous RAG maintains persistent retrieval indices that are incrementally updated, which could serve as a semantic pre-filter. CP's embedding-based variants act as lightweight proxies before LLM evaluation. But CP doesn't call this "pushdown" — it's more of a retrieval strategy choice in the dynamic planner. |

### 4.2.1 Vector-Semantic Gap

| Approach | Analysis |
|----------|----------|
| **Flink** | ❌ | 
| **Classical DB** | Limited — this is an inherent limitation of embedding-based retrieval. |
| **LOTUS** | ✅ Model cascade partially addresses this: embedding-based proxy catches easy cases, LLM handles hard cases where vectors are misleading. |
| **CP** | **Partial.** CP evaluates both embedding-based and LLM-based implementations for each operator and shows the accuracy gap. CP's dynamic planner can route uncertain cases (where embedding similarity is misleading) to LLM-based evaluation. But CP doesn't explicitly model the "semantic contradiction" failure mode. |

### 4.2.2-4 Graph-Level Problems (Triplestore Degradation, Graph-Agnostic Operators, N-Hop Myopia)

| Approach | Analysis |
|----------|----------|
| **Flink** | ❌ |
| **Classical DB** | Partial — PageRank, graph algorithms, N-hop traversal are classical graph DB techniques. |
| **LOTUS** | ❌ No graph semantics. |
| **CP** | ❌ **CP doesn't handle graph operations.** CP's operators work on flat tuple streams. Structural graph reasoning (community detection, N-hop traversal, topology-aware entity resolution) is entirely outside CP's scope. |

---

## Summary Matrix

> ✅ = directly helps, ◐ = partially helps, ❌ = doesn't help

| Problem | Flink | Classical DB | LOTUS | CP |
|---------|-------|-------------|-------|-----|
| **EverMemOS** | | | | |
| 1.1 Boundary detection latency | ✅ | ◐ (IVM) | ❌ | ✅ (semantic window) |
| 1.2 Unbounded profile state | ◐ | ✅ (IVM, hierarchical agg) | ❌ | ◐ (incremental agg) |
| 1.3 Multi-perspective redundancy | ◐ | ✅ (operator fusion) | ◐ | ✅ (operator fusion) |
| 1.4 Spiky latency | ✅ | ✅ (continuous agg) | ❌ | ✅ (incremental agg) |
| 1.5 Temporal dependencies | ◐ | ◐ (speculative exec) | ❌ | ◐ (embedding fallback) |
| **Mem0** | | | | |
| 2.1 Serial chaining | ◐ | ✅ (CBO, fusion) | ◐ (abstraction) | ✅ (operator fusion) |
| 2.2 1-to-N long tail | ✅ | ◐ (lateral join) | ◐ | ◐ (tuple batching) |
| 2.3 Top-1 retrieval risk | ❌ | ◐ (adaptive K) | ✅ (cascade) | ◐ |
| 2.4 2-stage redundancy | ❌ | ✅ (fusion, CSE) | ◐ | ✅ (map→map fusion) |
| **Zep/Graphiti** | | | | |
| 3.1 Join cost | ❌ | ✅ (semi-join) | ✅ (cascade) | ◐ (batching) |
| 3.2 Unconditional summary | ❌ | ✅ (lazy MV) | ◐ (sem_filter) | ✅ (filter→map fusion) |
| 3.3 Static communities | ◐ | ✅ (IVM, online clustering) | ❌ | ◐ (semantic group-by) |
| 3.4 Pairwise reduction | ◐ | ✅ (single-pass agg) | ✅ (sem_agg) | ✅ (incremental agg) |
| 3.5 N-to-1 race condition | ✅ | ✅ (GROUP BY + agg) | ✅ | ✅ (group-by + agg) |
| 3.6 Context fragmentation | ❌ | ◐ (affinity partition) | ❌ | ❌ |
| 3.7 Search amplification (3N) | ❌ | ✅ (query sharing) | ◐ | ◐ (batching) |
| 3.8 Fixed episode window | ✅ | ◐ (adaptive window) | ❌ | ✅ (semantic window) |
| **Cross-Cutting** | | | | |
| 4.1.1 Top-K cost explosion | ❌ | ◐ | ✅ (cascade) | ◐ |
| 4.1.2 Hard-coded workflows | ❌ | ✅ (CBO) | ◐ (abstraction) | ✅ (dynamic planner) |
| 4.1.3 Prompt bloat | ◐ | ✅ (IVM, delta) | ❌ | ◐ (rolling summary) |
| 4.1.4 Missing global context | ◐ | ✅ (CSE, self-join elim) | ◐ | ◐ (implicit via batching) |
| 4.1.5 Missing pushdown | ❌ | ✅ (predicate pushdown) | ✅ (semantic index) | ◐ (continuous RAG) |
| 4.2.1 Vector-semantic gap | ❌ | ❌ | ✅ (cascade) | ◐ (dynamic routing) |
| 4.2.2-4 Graph problems | ❌ | ◐ (graph algos) | ❌ | ❌ |

---

## Research Gap Analysis

**What CP adds over LOTUS**: CP fills LOTUS's biggest gap — **streaming support**. Where LOTUS is batch-only, CP provides continuous semantic operators with incremental state management. CP also goes further on **inter-operator optimization** (operator fusion with measured accuracy tradeoffs) and **dynamic plan adaptation** (MOBO-based runtime reconfiguration), which LOTUS lacks.

**What CP still lacks** (potential research opportunities):
1. **Graph-aware semantic operators**: Neither LOTUS nor CP handles graph topology (community detection, N-hop reasoning, topology-aware entity resolution). Agent memory systems heavily rely on graph structure.
2. **Full CBO with token-cost awareness**: CP's dynamic planner optimizes throughput-accuracy, but doesn't model LLM token costs ($/input_token, $/output_token). A true semantic CBO would include monetary cost as a first-class optimization objective.
3. **Model cascade**: LOTUS's cascade optimization (proxy model → gold model) is absent in CP. CP uses embedding-based variants as alternative implementations but doesn't chain them as a cascade with statistical accuracy guarantees.
4. **Cross-operator reordering**: CP fuses adjacent operators but doesn't reorder them. A full CBO would explore different operator orderings (e.g., push sem_filter before sem_map).
5. **Stateful prompt optimization**: Neither CP nor LOTUS explicitly optimize for LLM KV-cache prefix sharing or delta-based prompt construction for stateful read-modify-write patterns.
