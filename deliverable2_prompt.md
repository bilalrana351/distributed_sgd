You are a very professional report writer. You have written hundreds of system-design documents for real distributed systems courses and real engineering teams. You write in **simple, basic, to-the-point English** — no jargon padding, no filler, no marketing tone. Short sentences. Plain words. Where a technical term is unavoidable, define it in one line the first time it appears. Use tables and numbered lists wherever they make the content easier to read than a paragraph.

## What you are writing

I need you to produce **Deliverable 2: Assignment 2 — Full System Design Document** for my ongoing Parallel and Distributed Computing project. The project is already built and evaluated through Phases 1–5; this assignment is the **analytical system-design write-up** on top of that work.

**Scope clarification — very important.** This is the *design document* at **CLO 3 / C-5 (Evaluate)**. It is **not** the evaluation report. The full empirical evaluation (measured scalability, sensitivity sweep results, comparative baselines, measurable improvement, novelty justification) belongs to **Deliverable 3 (Final Presentation & Demonstration, C-6)**, which I will write separately. Therefore:

- D2 presents **models, predictions, assumptions, and reasoned trade-offs** — not full result tables.
- D2 is allowed to cite Phase 4/5 numbers **only as lightweight validation checks** for each analytical model (one or two numbers per section at most, each framed as "model predicts X; measurement is Y; deviation explained by Z"). 
- D2 must not reproduce `docs/phase4_results_tables.md` or `docs/phase5_per_seed_table.md` in full.
- Rigorous empirical evaluation (the big plots, the threshold-sweep analysis, the speedup calculations, the novelty claim) is **explicitly out of scope** for this document.

## Step 0 — Read these files before writing a single word

Do not start writing until you have read all of these.

**Proposal / problem context**
- `docs/project_context_simple.md`
- `docs/proposal_docs/pdc_section_01_cover_problem_challenge.md`
- `docs/proposal_docs/pdc_section_02_literature_review_and_table.md`
- `docs/proposal_docs/pdc_section_03_gap_hypothesis.md`
- `docs/proposal_docs/pdc_section_04_metrics_and_setup.md`
- `docs/proposal_docs/pdc_section_05_validation_and_references.md`

**Phase end-to-end records (architecture + what was actually built)**
- `docs/final_summary_docs_per_phase/phase1_end_to_end_record.md`
- `docs/final_summary_docs_per_phase/phase2_end_to_end_record.md`
- `docs/final_summary_docs_per_phase/phase3_end_to_end_record.md`
- `docs/final_summary_docs_per_phase/phase4_end_to_end_record.md`
- `docs/final_summary_docs_per_phase/phase5_end_to_end_record.md`

**Results evidence (read for orientation only — D2 cites these sparingly as validation)**
- `docs/phase4_results_tables.md`
- `docs/phase4_results_discussion.md`
- `docs/phase5_final_report.md`

**Implementation + infrastructure**
- `src/hybrid/` (controller, PS loop, launcher, metrics collector)
- `src/common/`, `src/sync_ddp/`, `src/async_ps/`
- `docker/Dockerfile`, `docker/docker-compose.yml`
- `requirements.txt`
- `configs/phase4_s{0,A,B}_{sync,async,hybrid}.yaml`

## Output: one file

Write the entire design document to **`docs/assignment2_system_design.md`**. Single Markdown file, ready for submission. At the top, include a small metadata block: title, course, CLO/PLO tags (CLO 3, PLO-4, C-5 Evaluate), and a one-sentence abstract. Add a one-line note that empirical evaluation lives in Deliverable 3.

## Required sections (all 8, in this order)

For every section, follow this internal shape: **(a) what the model/choice is**, **(b) why this choice given our problem**, **(c) what the trade-offs are**, **(d) one-line validation check against a Phase 4/5 number** — just enough to show the model is consistent with reality, not a full evaluation. Every section ends with a short **"Evaluation"** paragraph judging the design choice (strength, weakness, what C-5 "evaluate" means for this section).

### 1. System architecture diagram
- Mermaid `flowchart` diagram showing: Parameter Server process, N worker processes, gradient queue, shared model memory, `mode_flag`, `sync_round` counter, Controller module, CSV log sinks, and the Docker container boundary enclosing all of it.
- Second smaller Mermaid diagram showing the **control-signal path**: step-time samples → metrics collector (EMA) → controller decision → `mode_flag` flip → aggregator behaviour change.
- Follow both diagrams with a numbered component list (name, responsibility, language/library, in-memory or on-disk).

### 2. Communication model
- Classify every inter-component interaction by primitive: broadcast, scatter, gather, all-reduce, point-to-point.
- State the **transport** actually used (shared-memory queues + `multiprocessing.Lock`) and explain why this is a valid **emulation** of the proposal's container-based topology. Be honest about the Windows-CPU-forced transport choice.
- Two tables:
  1. Sync mode communication pattern per logical round.
  2. Async mode communication pattern per gradient push.
- Include a **message-count** column in each table (this becomes theoretical input to Section 5; D3 will measure actual counts).

### 3. Consistency model
- Use proper distributed-systems vocabulary: strong, eventual, bounded-staleness.
- Classify each mode: sync = strong (BSP-style); async = bounded staleness (bound = `param_pull_every`); hybrid = mode-dependent consistency that changes over time.
- Explain what a "consistency switch" means for the weights workers see when `mode_flag` flips.
- One-line validation: cite the staleness bound observed in Phase 4 (e.g., sA hybrid max staleness ≈ 400) to show the bound is respected. **Do not expand into full staleness tables — that is D3.**

### 4. Failure model
- Table of failure classes: worker crash, PS crash, stalled queue, deadlock at mode transition, lost gradient (queue full), indefinite straggler, controller thrashing.
- For each: detection mechanism (or "undetected"), current mitigation (cite cooldown + decision-window for controller thrashing), residual risk.
- Explicitly name failure classes the current system does **not** handle: Byzantine workers, inter-container network partition, disk-full on log sink, checkpoint corruption — and what would be needed to handle them.
- **This is the failure *model*. The failure *scenario evaluation* (actually running failures and measuring) belongs to D3 — do not do it here.**

### 5. Performance modeling
Include all three analytical tools, **as models** — not as result tables:

- **Amdahl's law** — identify the serial fraction `s` in sync mode (PS aggregation + broadcast). Derive a symbolic expression for speedup at N workers. Give a numeric bound for N=2, 4, 8 under a stated assumption for `s`.
- **Gustafson's law** — argue the weak-scaling case (global batch grows with N). Give the projection formula.
- **Message complexity** — derive closed-form expressions per logical round:
  - Sync PS-emulated: `M_sync(N)` — state whether the count matches ring-allreduce (`2N − 2`) or PS gather+broadcast (`2N`), and which one our implementation actually uses.
  - Async PS: `M_async(N) = 2` per worker update.
  - Hybrid: weighted expression `α · M_sync + (1 − α) · M_async` with α = sync fraction.
- End the section with **one short validation paragraph**: state that Phase 4 observed hybrid throughput ≈ 23.3 img/s on sA vs sync ≈ 13.6 img/s, and note whether that is *consistent with* the model's prediction. **One sentence, not a table. Full empirical speedup analysis is D3's job.**

### 6. Scalability assumptions
- Four axes: worker count (N), model size (parameters), dataset size (samples), cluster heterogeneity (fraction of stragglers).
- For each: the **assumption** (e.g., "shared-memory queue scales cleanly to ~8 workers on a single host; beyond that, TCP or RPC transport is required"), the **expected limit**, the **breaking point**.
- One table. Each row = one axis.
- One paragraph on what code changes would move the system from the current 1-container-N-processes emulation to a true multi-container `torch.distributed` (gloo) deployment.
- **These are *assumptions* — the *measured* scalability curves belong to D3.**

### 7. Implementation plan
- Three timelines:
  1. **Already done** — map Phases 1–5 to folders/files (1–2 sentences per phase; reuse the phase records).
  2. **Near-term, low-risk** — apply best sweep config (`svh=0.25, gvl=0.70, cd=100`) to Scenario B. Concrete tasks: YAML edit → config path → run command → expected artefact.
  3. **Longer-term** — migrate to multi-container `torch.distributed` gloo on a Docker bridge network; extend to CIFAR-100 + MobileNetV2. WBS-style list, effort as S/M/L.
- Responsibility matrix (even for solo project: Implementation / Testing / Documentation / Presentation).
- Simple Gantt-style table with week numbers.
- Mention Docker reproducibility (`docker/Dockerfile` + `docker/docker-compose.yml`) as the deployment artefact.

### 8. Risk analysis
- Probability × impact table (Low / Medium / High on each axis, risk level printed as `[LOW]` / `[MED]` / `[HIGH]`).
- Must include: controller over-switching under transient stragglers, CPU-only throughput ceiling, mixed log vintage bias, Docker image drift (check whether `requirements.txt` pins versions — flag if not), checkpoint bloat during sweeps, reproducibility risk if `threshold_folders/` is lost.
- For each risk: likelihood, impact, current mitigation, proposed mitigation, residual risk.
- End with a top-3 risks summary paragraph.

## Writing-style rules

1. Simple English. Short sentences.
2. Tables over paragraphs wherever comparison is happening.
3. Numbers over adjectives.
4. Honest about limitations (Windows-CPU, 2 workers, mixed log vintage).
5. Every section ends with a short **Evaluation** paragraph — this is what makes the document C-5 rather than C-3.
6. Do not reproduce full result tables from Phase 4/5. At most one number per section as a sanity check for the model being presented.
7. Do not invent experiments or numbers.

## Final checks

- [ ] All 8 sections present, in order.
- [ ] Mermaid diagrams render inline.
- [ ] Each section has an Evaluation paragraph.
- [ ] Empirical citations are one-line sanity checks, not result dumps.
- [ ] Docker reproducibility mentioned in Sections 1 and 7.
- [ ] File written to `docs/assignment2_system_design.md`.

Do not ask me for permission before writing files. When done, print a short summary of what was written, which sections use which analytical model, and which one-line empirical validation was cited in each.