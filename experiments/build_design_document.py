"""
Deliverable 2 — Assignment 2 Full System Design Document (builder).

Generates a submission-ready Microsoft Word (.docx) file at:
    docs/assignment2_system_design.docx

The document is written as a *forward-looking system design* — it describes
the planned architecture, communication model, consistency contract, failure
model, performance model, scalability assumptions, implementation plan, and
risk analysis for the hybrid distributed SGD project proposed in
Deliverable 1. Empirical evaluation and measured results are explicitly
reserved for Deliverable 3 (Final Presentation and Demonstration).

Run from the repository root::

    .\\.venv\\Scripts\\python.exe experiments\\build_design_document.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


# --------------------------------------------------------------------------- #
#  Paths                                                                       #
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCX_OUT = REPO_ROOT / "docs" / "assignment2_system_design.docx"

NAVY = RGBColor(0x0E, 0x1E, 0x3C)
STEEL = RGBColor(0x1F, 0x3A, 0x5F)
SLATE = RGBColor(0x55, 0x66, 0x77)
MUTED = RGBColor(0x77, 0x77, 0x77)


# --------------------------------------------------------------------------- #
#  XML / styling helpers                                                       #
# --------------------------------------------------------------------------- #


def _set_cell_shading(cell, hex_color: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tc_pr.append(shd)


def _set_cell_borders(cell, color_hex: str = "BFBFBF") -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_borders = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        border = OxmlElement(f"w:{edge}")
        border.set(qn("w:val"), "single")
        border.set(qn("w:sz"), "4")
        border.set(qn("w:color"), color_hex)
        tc_borders.append(border)
    tc_pr.append(tc_borders)


def configure_styles(doc: Document) -> None:
    """Apply a clean, professional look to the default document styles."""
    styles = doc.styles

    normal = styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.15

    for level, size in ((1, 18), (2, 14), (3, 12)):
        style = styles[f"Heading {level}"]
        style.font.name = "Calibri"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = STEEL
        style.paragraph_format.space_before = Pt(14)
        style.paragraph_format.space_after = Pt(4)

    title_style = styles["Title"]
    title_style.font.name = "Calibri"
    title_style.font.size = Pt(26)
    title_style.font.bold = True
    title_style.font.color.rgb = NAVY

    subtitle_style = styles["Subtitle"]
    subtitle_style.font.name = "Calibri"
    subtitle_style.font.size = Pt(14)
    subtitle_style.font.italic = True
    subtitle_style.font.color.rgb = SLATE


# --------------------------------------------------------------------------- #
#  Paragraph / block builders                                                  #
# --------------------------------------------------------------------------- #


def add_para(
    doc: Document,
    text: str,
    bold: bool = False,
    italic: bool = False,
    align: int | None = None,
) -> None:
    para = doc.add_paragraph()
    if align is not None:
        para.alignment = align
    run = para.add_run(text)
    run.bold = bold
    run.italic = italic


def add_heading(doc: Document, text: str, level: int) -> None:
    doc.add_heading(text, level=level)


def add_bullets(doc: Document, items: Iterable[str]) -> None:
    for item in items:
        doc.add_paragraph(item, style="List Bullet")


def add_numbered(doc: Document, items: Iterable[str]) -> None:
    for item in items:
        doc.add_paragraph(item, style="List Number")


def add_code_block(doc: Document, text: str, caption: str | None = None) -> None:
    """Render a fenced code / diagram source block as a shaded single-cell table."""
    if caption:
        cap = doc.add_paragraph()
        cap_run = cap.add_run(caption)
        cap_run.italic = True
        cap_run.font.size = Pt(10)
        cap_run.font.color.rgb = MUTED

    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    cell = table.rows[0].cells[0]
    _set_cell_shading(cell, "F4F4F4")
    _set_cell_borders(cell, "DDDDDD")
    cell.text = ""

    lines = text.splitlines() or [""]
    for i, line in enumerate(lines):
        para = cell.paragraphs[0] if i == 0 else cell.add_paragraph()
        para.paragraph_format.space_after = Pt(0)
        run = para.add_run(line)
        run.font.name = "Consolas"
        run.font.size = Pt(9)

    doc.add_paragraph()


def add_table(
    doc: Document,
    headers: List[str],
    rows: List[List[str]],
    col_widths_cm: List[float] | None = None,
) -> None:
    """Insert a cleanly styled table with a navy header row."""
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.LEFT

    # Header
    header_cells = table.rows[0].cells
    for idx, text in enumerate(headers):
        cell = header_cells[idx]
        cell.text = ""
        para = cell.paragraphs[0]
        run = para.add_run(text)
        run.bold = True
        run.font.size = Pt(10)
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        _set_cell_shading(cell, "1F3A5F")
        _set_cell_borders(cell, "1F3A5F")
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER

    # Body
    for r_idx, row in enumerate(rows, start=1):
        zebra = r_idx % 2 == 0
        for c_idx, cell_text in enumerate(row):
            cell = table.rows[r_idx].cells[c_idx]
            cell.text = ""
            para = cell.paragraphs[0]
            run = para.add_run(cell_text)
            run.font.size = Pt(10)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            if zebra:
                _set_cell_shading(cell, "F6F8FB")
            _set_cell_borders(cell)

    if col_widths_cm is not None:
        for idx, width in enumerate(col_widths_cm):
            for row in table.rows:
                row.cells[idx].width = Cm(width)

    doc.add_paragraph()


def add_evaluation(doc: Document, text: str) -> None:
    """Closing 'Evaluation' paragraph that justifies C-5 (Evaluate) depth."""
    para = doc.add_paragraph()
    label = para.add_run("Evaluation. ")
    label.bold = True
    label.font.color.rgb = STEEL
    body = para.add_run(text)
    body.italic = False


def add_page_break(doc: Document) -> None:
    doc.add_page_break()


# --------------------------------------------------------------------------- #
#  Mermaid diagram sources (rendered as monospace code blocks inside docx)    #
# --------------------------------------------------------------------------- #

MERMAID_MAIN_ARCH = """flowchart LR
    subgraph CONTAINER["Container boundary (reproducible image)"]
        direction LR

        subgraph PS["Parameter Server (rank 0)"]
            AGG["Aggregator\\n(sync average OR async apply)"]
            CTRL["Controller\\n(thresholds + cooldown + window)"]
            MC["Metrics Collector\\n(EMA of runtime signals)"]
            MODEL[("Shared model memory\\n(global parameters)")]
            FLAG[["mode_flag\\n(0 = sync, 1 = async)"]]
            SR[["sync_round counter"]]
        end

        W1[["Worker 1"]]
        W2[["Worker 2"]]
        WN[["Worker N"]]

        GQ{{"Gradient queue"}}
        VQ{{"Validation queue"}}
        LOCK[/"Parameter lock"/]

        CSV[(CSV log sinks:\\nPS summary, switches,\\nstaleness, per-worker)]
        CKPT[(Model checkpoints)]
    end

    W1 -- "gradient tuple" --> GQ
    W2 -- "gradient tuple" --> GQ
    WN -- "gradient tuple" --> GQ
    GQ --> AGG
    AGG -- "read / write under LOCK" --> MODEL
    AGG -- "increment on sync round" --> SR
    W1 -. "pull parameters" .-> MODEL
    W2 -. "pull parameters" .-> MODEL
    WN -. "pull parameters" .-> MODEL
    W1 --> VQ
    VQ --> AGG
    MC --> CTRL
    CTRL -- "flip" --> FLAG
    FLAG -.-> AGG
    AGG --> CSV
    CTRL --> CSV
    AGG --> CKPT
"""

MERMAID_CONTROL_SIGNAL = """flowchart LR
    ST["Per-worker step-time samples\\nGradient-norm samples\\nLoss samples"]
    EMA["Metrics Collector\\n(EMA smoothing)"]
    SIG["Signal vector\\n{ worker_time_var,\\n  grad_norm_var,\\n  loss_trend }"]
    DEC["Controller.evaluate()\\nthresholds + cooldown\\n+ decision window"]
    FLIP{{"mode_flag flip?"}}
    AGGB["Aggregator behaviour change\\n(sync average ↔ async apply)"]
    LOG[("Switch log entry")]

    ST --> EMA --> SIG --> DEC --> FLIP
    FLIP -- yes --> AGGB
    FLIP -- yes --> LOG
    FLIP -- no --> DEC
"""


# --------------------------------------------------------------------------- #
#  Cover page                                                                  #
# --------------------------------------------------------------------------- #


def build_cover(doc: Document) -> None:
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("Hybrid Distributed SGD")
    run.font.name = "Calibri"
    run.font.size = Pt(30)
    run.font.bold = True
    run.font.color.rgb = NAVY

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub_run = sub.add_run(
        "System Design Document — Parameter Server and Ring All-Reduce "
        "with a Bi-Directional Online Controller"
    )
    sub_run.italic = True
    sub_run.font.size = Pt(13)
    sub_run.font.color.rgb = SLATE

    doc.add_paragraph()

    deliverable = doc.add_paragraph()
    deliverable.alignment = WD_ALIGN_PARAGRAPH.CENTER
    drun = deliverable.add_run("Deliverable 2 — Assignment 2 Full System Design")
    drun.bold = True
    drun.font.size = Pt(14)
    drun.font.color.rgb = STEEL

    doc.add_paragraph()
    doc.add_paragraph()

    add_table(
        doc,
        headers=["Field", "Value"],
        rows=[
            ["Course", "Parallel and Distributed Computing (Spring 2026)"],
            ["Department", "School of Electrical Engineering and Computer Sciences (SEECS)"],
            ["Institution", "National University of Sciences and Technology (NUST), Islamabad"],
            ["Programme / Section", "BS Computer Science — Section 13-E — Semester 6"],
            ["Deliverable", "D2: Full System Design Document"],
            ["Learning outcome", "CLO 3 · PLO-4 · C-5 (Evaluate)"],
            [
                "Assessment focus",
                "Analytical depth: models, trade-offs, and reasoned design "
                "judgements over a planned distributed system.",
            ],
            [
                "Companion deliverables",
                "D1 Project Proposal (submitted) · D3 Final Presentation and "
                "Demonstration (C-6, separate submission)",
            ],
        ],
        col_widths_cm=[4.5, 12.0],
    )

    doc.add_paragraph()
    team_heading = doc.add_paragraph()
    team_heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
    th_run = team_heading.add_run("Project Team")
    th_run.bold = True
    th_run.font.size = Pt(13)
    th_run.font.color.rgb = STEEL

    add_table(
        doc,
        headers=["Name", "Role"],
        rows=[
            ["Muhammad Haseeb Ul Haq", "Team Member"],
            ["Muhammad Samama Usman", "Team Member"],
            ["Bilal Rana", "Team Member"],
            ["Muhammad Moiz", "Team Member"],
        ],
        col_widths_cm=[9.0, 7.5],
    )

    doc.add_paragraph()
    abs_para = doc.add_paragraph()
    abs_run_label = abs_para.add_run("Abstract. ")
    abs_run_label.bold = True
    abs_run_label.font.color.rgb = STEEL
    abs_run_body = abs_para.add_run(
        "This document presents the full system design of a hybrid "
        "distributed Stochastic Gradient Descent (SGD) training system that "
        "alternates between synchronous Ring All-Reduce semantics and "
        "asynchronous Parameter Server semantics under the guidance of a "
        "lightweight, bi-directional, online controller. The design covers "
        "the architecture, communication primitives, consistency guarantees, "
        "failure classes, analytical performance bounds, scalability "
        "assumptions, execution plan, and risk register required at the "
        "C-5 (Evaluate) level. Empirical evaluation — measured speedups, "
        "sensitivity sweeps, and comparative baselines — is reserved for "
        "Deliverable 3."
    )

    scope_para = doc.add_paragraph()
    scope_label = scope_para.add_run("Scope note. ")
    scope_label.bold = True
    scope_label.font.color.rgb = STEEL
    scope_para.add_run(
        "This document is analytical and predictive. It defines the models "
        "the system is designed against, justifies each model choice, and "
        "states the trade-offs the design accepts. Concrete measurements "
        "from the running system are the subject of the evaluation report "
        "(Deliverable 3) and are therefore intentionally absent here."
    )

    add_page_break(doc)


# --------------------------------------------------------------------------- #
#  Table of contents and executive summary                                     #
# --------------------------------------------------------------------------- #


def build_toc(doc: Document) -> None:
    add_heading(doc, "Table of Contents", level=1)
    add_numbered(
        doc,
        [
            "Executive Summary and Problem Context",
            "System Architecture Diagram",
            "Communication Model",
            "Consistency Model",
            "Failure Model",
            "Performance Modelling (Amdahl, Gustafson, Message Complexity)",
            "Scalability Assumptions",
            "Implementation Plan",
            "Risk Analysis",
            "Conclusion and C-5 Summary",
        ],
    )
    add_page_break(doc)


def build_executive_summary(doc: Document) -> None:
    add_heading(doc, "Executive Summary and Problem Context", level=1)

    add_para(
        doc,
        "Modern deep learning models are too large to train on a single "
        "machine, which forces training to be distributed across many "
        "workers. When workers cooperate on the same model, the cost of "
        "sharing gradient updates often dominates the total training time. "
        "Two classical architectures address this problem in opposite ways. "
        "A synchronous Ring All-Reduce architecture enforces strong "
        "consistency by requiring every worker to participate in every "
        "update, but it is vulnerable to slow nodes (stragglers). An "
        "asynchronous Parameter Server architecture removes that wait and "
        "scales more gracefully under heterogeneous loads, but updates can "
        "become stale and final model accuracy can suffer.",
    )

    add_para(
        doc,
        "Real clusters rarely satisfy the assumptions behind either pure "
        "strategy. Latency spikes appear and disappear, workers occasionally "
        "slow down for short periods, and training itself passes through "
        "stable and unstable phases. Existing hybrid proposals usually "
        "switch in one direction only or assume that slow workers remain "
        "slow for the entire run. The design presented in this document "
        "addresses this gap.",
    )

    add_heading(doc, "Design thesis", level=2)
    add_bullets(
        doc,
        [
            "The system will run a single training job across one Parameter "
            "Server and N worker processes.",
            "The aggregation rule will switch at runtime between a "
            "synchronous averaged-gradient mode and an asynchronous "
            "per-gradient mode.",
            "A dedicated controller will observe three live signals — "
            "worker step-time variance, gradient-norm variance, and loss "
            "trend — and decide when to switch modes, subject to cooldown "
            "and hysteresis.",
            "All mode transitions, with their triggering reasons, will be "
            "logged so that behaviour is reviewable after each run.",
            "The entire environment will be containerised so that every "
            "component can be reproduced from a clean state.",
        ],
    )

    add_heading(doc, "Primary objectives", level=2)
    add_table(
        doc,
        headers=["Objective", "Design commitment"],
        rows=[
            [
                "Reduce time-to-convergence under stragglers",
                "Controller switches to asynchronous aggregation when "
                "persistent or transient slowness is detected.",
            ],
            [
                "Preserve final accuracy close to synchronous baseline",
                "Controller returns to synchronous aggregation when the "
                "training signal becomes unstable or stragglers resolve.",
            ],
            [
                "Operate safely on constrained hardware",
                "Bounded gradient queue, parameter lock, cooldown-based "
                "transitions, and explicit decision window to prevent "
                "oscillation.",
            ],
            [
                "Remain reproducible across machines",
                "Containerised runtime with pinned dependencies and "
                "bind-mounted output directories.",
            ],
        ],
        col_widths_cm=[6.5, 10.0],
    )

    add_evaluation(
        doc,
        "The problem statement establishes the two failure modes of the "
        "classical approaches (straggler vulnerability for synchronous; "
        "staleness for asynchronous) and positions the hybrid design as a "
        "dynamic policy over those two regimes rather than a new algorithm. "
        "The principal strength of the framing is that every design choice "
        "downstream can be justified by reference to one of these two "
        "failure modes. The principal limitation is that the framing assumes "
        "the controller has access to clean, low-latency signals; this "
        "assumption is made explicit in Section 6.",
    )
    add_page_break(doc)


# --------------------------------------------------------------------------- #
#  Section 1: Architecture                                                     #
# --------------------------------------------------------------------------- #


def build_section_architecture(doc: Document) -> None:
    add_heading(doc, "1. System Architecture Diagram", level=1)

    add_para(
        doc,
        "The system is a single-container, (1 + N) process layout. One "
        "process plays the role of Parameter Server and owns the global "
        "model. N processes play the role of workers, each holding a local "
        "copy of the model for forward and backward computation. All "
        "processes share a single set of model parameters in operating-"
        "system shared memory and communicate through multiprocessing "
        "queues and cross-process synchronisation primitives. The entire "
        "topology is enclosed within one reproducible container image.",
    )

    add_heading(doc, "1.1 Main architecture diagram", level=2)
    add_para(
        doc,
        "The following Mermaid diagram shows the process-level view of the "
        "system. A grader can paste the source into any Mermaid-capable "
        "viewer (for example, Mermaid Live Editor) to render it.",
    )
    add_code_block(doc, MERMAID_MAIN_ARCH, caption="Mermaid source — main architecture")

    add_heading(doc, "1.2 Control-signal path", level=2)
    add_para(
        doc,
        "The second diagram isolates the control path: how runtime "
        "measurements reach the controller, how the controller decides "
        "whether to change mode, and how that decision propagates back to "
        "the aggregator.",
    )
    add_code_block(
        doc,
        MERMAID_CONTROL_SIGNAL,
        caption="Mermaid source — control-signal path",
    )

    add_heading(doc, "1.3 Component list", level=2)
    add_table(
        doc,
        headers=["#", "Component", "Responsibility", "Language / Library", "Location"],
        rows=[
            [
                "1",
                "Parameter Server process",
                "Owns the global model; hosts the aggregator, the "
                "controller, and the metrics collector.",
                "Python 3.12 + PyTorch",
                "In-memory (shared)",
            ],
            [
                "2",
                "Worker processes (1..N)",
                "Perform forward and backward passes on a local data shard; "
                "push gradients; pull parameters.",
                "Python 3.12 + PyTorch",
                "In-memory (shared)",
            ],
            [
                "3",
                "Gradient queue",
                "FIFO channel carrying gradient tuples from workers to the "
                "server.",
                "Multiprocessing queue",
                "In-memory",
            ],
            [
                "4",
                "Validation queue",
                "Channel carrying periodic validation results from the "
                "evaluator worker back to the server for logging.",
                "Multiprocessing queue",
                "In-memory",
            ],
            [
                "5",
                "Shared model memory",
                "Single authoritative copy of the model parameters, "
                "accessible to every process.",
                "Shared-memory tensor",
                "In-memory (shared)",
            ],
            [
                "6",
                "Parameter lock",
                "Guards parameter reads and writes so that aggregation and "
                "worker pulls never observe partial updates.",
                "Multiprocessing lock",
                "In-memory",
            ],
            [
                "7",
                "Mode flag",
                "Shared integer (0 = synchronous, 1 = asynchronous); read "
                "by the aggregator every round; written by the controller.",
                "Multiprocessing value",
                "In-memory",
            ],
            [
                "8",
                "Sync-round counter",
                "Monotonic counter incremented at the end of each "
                "synchronous round; workers wait on it.",
                "Multiprocessing value",
                "In-memory",
            ],
            [
                "9",
                "Controller module",
                "Threshold-based policy with cooldown and decision window; "
                "decides when to flip the mode flag.",
                "Pure Python",
                "In-memory",
            ],
            [
                "10",
                "Metrics Collector",
                "Maintains exponential moving averages of worker time "
                "variance, gradient-norm variance, and loss trend.",
                "Pure Python",
                "In-memory",
            ],
            [
                "11",
                "Structured log sinks",
                "Time-indexed CSV streams for server summary, mode "
                "switches, staleness samples, and per-worker training "
                "records.",
                "Standard CSV writer",
                "On-disk (bind mount)",
            ],
            [
                "12",
                "Checkpoint sink",
                "Persistent snapshots of the model, global step, and "
                "switch history for post-run inspection and recovery.",
                "PyTorch serialisation",
                "On-disk (bind mount)",
            ],
            [
                "13",
                "Container runtime",
                "Reproducible execution environment hosting the entire "
                "topology; exposes only the bind-mounted output directories.",
                "Docker (Linux image)",
                "Container image",
            ],
        ],
        col_widths_cm=[0.9, 3.8, 6.0, 3.2, 2.6],
    )

    add_evaluation(
        doc,
        "The architecture is organised around exactly one control point — "
        "the Parameter Server — and exactly one mutable shared object per "
        "concern (mode flag, sync-round counter, gradient queue, parameter "
        "lock). That discipline keeps reasoning about consistency and "
        "failure straightforward. The main limitation is that the entire "
        "topology sits in one process tree within one container; this is an "
        "emulation of the multi-container target described in the project "
        "proposal rather than the target itself. The design preserves the "
        "training semantics of both synchronous and asynchronous modes, so "
        "transitioning to a true multi-container deployment is a transport-"
        "layer change rather than a redesign.",
    )
    add_page_break(doc)


# --------------------------------------------------------------------------- #
#  Section 2: Communication Model                                              #
# --------------------------------------------------------------------------- #


def build_section_communication(doc: Document) -> None:
    add_heading(doc, "2. Communication Model", level=1)

    add_para(
        doc,
        "Every inter-component interaction in the system is classified here "
        "by its underlying distributed-systems primitive (broadcast, "
        "scatter, gather, reduce, point-to-point, all-reduce). Because the "
        "planned execution environment places all processes inside one "
        "container, the transport layer is a set of in-process shared "
        "structures (a gradient queue and a parameter lock) rather than a "
        "network stack. This choice is an honest emulation of the "
        "proposal's container-based topology: the semantics of every "
        "primitive are preserved, and only the wire is replaced.",
    )

    add_heading(doc, "2.1 Transport choice and justification", level=2)
    add_bullets(
        doc,
        [
            "The proposal envisaged a PyTorch RPC backend for the "
            "asynchronous parameter server and a PyTorch Distributed "
            "(Gloo) backend for the synchronous All-Reduce side.",
            "On the Windows, CPU-only development environment available "
            "for this study, the RPC backend is not available, and "
            "mid-run backend switching between two distinct process groups "
            "is non-trivial.",
            "The design therefore commits to a single transport "
            "(multiprocessing with shared memory) for both modes and "
            "changes only the aggregator's rule when the mode flag flips.",
            "Because the per-mode semantics (averaged-gradient rounds "
            "under synchronous mode; immediate per-gradient updates under "
            "asynchronous mode) are preserved, migration to a multi-"
            "container Gloo deployment is planned as a transport swap and "
            "is discussed in Section 6.",
        ],
    )

    add_heading(doc, "2.2 Synchronous mode — communication per logical round", level=2)
    add_table(
        doc,
        headers=[
            "Step",
            "Primitive",
            "From → To",
            "Payload",
            "Transport",
            "Msg count",
        ],
        rows=[
            [
                "1",
                "Point-to-point",
                "Worker i → Server",
                "Gradient tuple (gradients, step, loss, timing, norm).",
                "Shared-memory queue",
                "N",
            ],
            [
                "2",
                "Gather (logical)",
                "Server-internal",
                "Server buffers one gradient per worker until the round is "
                "complete.",
                "In-process buffer",
                "0",
            ],
            [
                "3",
                "Reduce (average)",
                "Server-internal",
                "Averaged gradient applied as a single update under the "
                "parameter lock.",
                "In-process",
                "0",
            ],
            [
                "4",
                "Broadcast (emulated)",
                "Server → Workers",
                "Sync-round counter is incremented; workers read shared "
                "parameters under the lock.",
                "Shared memory + counter",
                "1",
            ],
        ],
        col_widths_cm=[1.2, 2.6, 3.0, 5.2, 2.9, 1.6],
    )
    add_para(
        doc,
        "Total messages per synchronous round: N + 1 (N worker pushes plus "
        "a single broadcast-equivalent counter increment that releases all "
        "workers).",
        italic=True,
    )

    add_heading(doc, "2.3 Asynchronous mode — communication per gradient push", level=2)
    add_table(
        doc,
        headers=[
            "Step",
            "Primitive",
            "From → To",
            "Payload",
            "Transport",
            "Msg count",
        ],
        rows=[
            [
                "1",
                "Point-to-point",
                "Worker i → Server",
                "Gradient tuple (same schema as synchronous mode).",
                "Shared-memory queue",
                "1",
            ],
            [
                "2",
                "Reduce (apply)",
                "Server-internal",
                "Gradient applied immediately under the parameter lock.",
                "In-process",
                "0",
            ],
            [
                "3",
                "Pull (periodic)",
                "Worker i ← Server",
                "Local model refreshed from shared parameters every K "
                "local steps, where K is a configuration parameter.",
                "Shared memory",
                "1 / K",
            ],
        ],
        col_widths_cm=[1.2, 2.6, 3.0, 5.2, 2.9, 1.6],
    )
    add_para(
        doc,
        "Total messages per asynchronous update: approximately 1 + 1/K per "
        "worker step (one gradient push plus an amortised periodic pull).",
        italic=True,
    )

    add_heading(doc, "2.4 Why this is a faithful emulation", level=2)
    add_para(
        doc,
        "Each message in the two tables has a direct network equivalent. "
        "A worker push maps to an RPC send; an emulated broadcast through "
        "a sync-round counter and shared memory maps to a parameter "
        "refresh over RPC or to an All-Reduce completion callback over "
        "Gloo. Because the controller, the queue, the parameter lock, and "
        "the staleness semantics are all written against these primitives "
        "rather than against a specific transport, the design is a genuine "
        "emulation of the proposal's topology at the primitive level.",
    )

    add_heading(doc, "2.5 Predicted behaviour against the message model", level=2)
    add_para(
        doc,
        "The communication model predicts that hybrid-mode wall-clock "
        "throughput should fall between the pure synchronous and pure "
        "asynchronous bounds, weighted by the fraction of time the "
        "controller spends in each regime. That weighting appears again "
        "as the mixing coefficient in the performance model of Section 5. "
        "Concrete per-run message counts and the resulting throughput are "
        "scheduled for measurement in Deliverable 3.",
    )

    add_evaluation(
        doc,
        "The communication model matches the proposal's primitives exactly "
        "(point-to-point, gather, reduce, broadcast), and the synchronous "
        "message count of N + 1 is the classical Parameter Server gather-"
        "plus-broadcast cost. This is a known upper bound on the "
        "bandwidth-optimal ring-allreduce count of 2(N − 1), so the design "
        "deliberately trades a small amount of bandwidth efficiency for a "
        "single aggregation point that simplifies controller reasoning. "
        "The trade is sound for small N on one host; it must be revisited "
        "when moving to a true multi-container deployment, at which point "
        "the PS gather pattern should be replaced by native All-Reduce.",
    )
    add_page_break(doc)


# --------------------------------------------------------------------------- #
#  Section 3: Consistency Model                                                #
# --------------------------------------------------------------------------- #


def build_section_consistency(doc: Document) -> None:
    add_heading(doc, "3. Consistency Model", level=1)

    add_para(
        doc,
        "The consistency contract describes which view of the model "
        "parameters each worker is guaranteed to see. Because the hybrid "
        "system alternates between two aggregation regimes, the "
        "consistency guarantee is mode-dependent and must be stated "
        "precisely for each regime.",
    )

    add_heading(doc, "3.1 Definitions", level=2)
    add_table(
        doc,
        headers=["Term", "One-line definition in this design"],
        rows=[
            [
                "Strong consistency",
                "Every worker observes the same parameters at the same "
                "logical step — the Bulk Synchronous Parallel (BSP) "
                "guarantee.",
            ],
            [
                "Eventual consistency",
                "Workers may temporarily observe stale parameters; given "
                "enough updates, all workers converge to a common state; "
                "no bound on staleness.",
            ],
            [
                "Bounded staleness",
                "Workers may observe stale parameters, but any view is "
                "guaranteed to be no more than K local steps out of date, "
                "where K is a fixed configuration parameter.",
            ],
        ],
        col_widths_cm=[4.5, 12.0],
    )

    add_heading(doc, "3.2 Per-mode classification", level=2)
    add_table(
        doc,
        headers=["Mode", "Consistency class", "Bound", "What workers observe"],
        rows=[
            [
                "Synchronous",
                "Strong",
                "0 by construction",
                "All N workers pull the same parameters after every "
                "averaged update; no staleness.",
            ],
            [
                "Asynchronous",
                "Bounded staleness",
                "K local steps (pull interval)",
                "A worker may push gradients computed on parameters that "
                "are at most K steps behind the global state.",
            ],
            [
                "Hybrid",
                "Mode-dependent",
                "0 during synchronous segments; K during asynchronous "
                "segments",
                "Consistency class changes precisely at each mode-flag "
                "transition.",
            ],
        ],
        col_widths_cm=[3.0, 3.5, 4.0, 6.0],
    )

    add_heading(doc, "3.3 Semantics of a consistency switch", level=2)
    add_para(
        doc,
        "When the controller switches from synchronous to asynchronous at "
        "global step g, the parameters at step g are the last strongly "
        "consistent snapshot. From step g + 1 onward, workers observe "
        "bounded-staleness views. When the controller switches back from "
        "asynchronous to synchronous at step g', the server buffers the "
        "next N gradients and performs one averaged update before "
        "releasing workers; after that release, the system is again "
        "strongly consistent. Crucially, no parameter is ever rolled back: "
        "staleness only grows during asynchronous segments and is capped "
        "at K.",
    )

    add_heading(doc, "3.4 Planned staleness bound", level=2)
    add_para(
        doc,
        "The design targets an observed maximum staleness that scales with "
        "the product of the asynchronous-segment length and the worker "
        "step count. With the nominal pull interval K in the low tens of "
        "local steps, staleness is predicted to remain in the hundreds of "
        "global steps for laptop-scale runs. Measured staleness "
        "distributions, including percentile bands and mode-aligned "
        "timelines, are planned for Deliverable 3.",
    )

    add_evaluation(
        doc,
        "The hybrid system offers a controller-driven dial between "
        "statistical quality (strong consistency, sync mode) and hardware "
        "utilisation (bounded staleness, async mode) — a distinctive "
        "property over the single-regime baselines in the literature. The "
        "honest weakness is that a consistency switch is not free: the "
        "first synchronous aggregation after an asynchronous segment "
        "merges temporarily stale gradients with a fresh averaged step, "
        "which can perturb the loss trajectory. The design contains this "
        "risk by requiring a cooldown and a decision window before any "
        "mode flip; nevertheless, the semantics of the transition remain "
        "the most subtle part of the contract and are explicitly listed "
        "as a tuning target in Section 8.",
    )
    add_page_break(doc)


# --------------------------------------------------------------------------- #
#  Section 4: Failure Model                                                    #
# --------------------------------------------------------------------------- #


def build_section_failure(doc: Document) -> None:
    add_heading(doc, "4. Failure Model", level=1)

    add_para(
        doc,
        "The failure model enumerates the failure classes the design "
        "addresses, the detection mechanism for each, the mitigation "
        "provided, and the residual risk after mitigation. The section "
        "also lists failure classes that are deliberately out of scope "
        "for the current design, along with what would be required to "
        "handle them. This is the failure model; empirical fault-"
        "injection experiments are scheduled for Deliverable 3.",
    )

    add_heading(doc, "4.1 Failure classes addressed by the design", level=2)
    add_table(
        doc,
        headers=[
            "Failure class",
            "Detection",
            "Mitigation (planned)",
            "Residual risk",
        ],
        rows=[
            [
                "Worker crash",
                "Implicit: gradient queue stops receiving from the failed "
                "worker.",
                "Run exits once all workers terminate or the server is "
                "interrupted; checkpoint allows resumption.",
                "High — single worker loss silently reduces effective N.",
            ],
            [
                "Parameter server crash",
                "Implicit: workers block on the gradient queue when it is "
                "full.",
                "External kill and restart from the latest checkpoint.",
                "High — there is no watchdog or leader re-election.",
            ],
            [
                "Stalled gradient queue (back-pressure)",
                "Bounded queue size plus server-side poll timeout.",
                "Workers block on push; no gradient is silently dropped.",
                "Low, at the cost of additional worker latency.",
            ],
            [
                "Deadlock at mode transition",
                "Server logs every flip; transition handler explicitly "
                "releases any worker blocked on the sync-round counter.",
                "Single counter increment on the transition path "
                "unblocks synchronous-mode waiters.",
                "Low — covered by design.",
            ],
            [
                "Lost gradient (queue capacity)",
                "Bounded queue size plus poll timeout.",
                "Blocking semantics are preferred over silent drop.",
                "Medium — severe stragglers may convert the block into a "
                "latency bottleneck.",
            ],
            [
                "Indefinite straggler",
                "Controller signal (worker-time variance) crosses the "
                "configured high threshold.",
                "Switch to asynchronous mode after cooldown and decision "
                "window.",
                "Medium — the asynchronous segment can persist if the "
                "straggler never recovers.",
            ],
            [
                "Controller thrashing",
                "Controller tracks consecutive triggers and time-in-mode.",
                "Mandatory cooldown period and decision window; "
                "exponential moving averages smooth all input signals.",
                "Low on persistent scenarios; medium on transient "
                "scenarios if thresholds are mis-tuned.",
            ],
        ],
        col_widths_cm=[3.0, 3.5, 5.0, 4.0],
    )

    add_heading(doc, "4.2 Failure classes intentionally out of scope", level=2)
    add_table(
        doc,
        headers=["Failure class", "What would be required"],
        rows=[
            [
                "Byzantine workers (malicious or corrupted gradients)",
                "Per-worker gradient sanitisation beyond norm clipping; "
                "robust aggregators (Krum, trimmed mean, median); "
                "cryptographic signing of worker pushes.",
            ],
            [
                "Inter-container network partition",
                "Requires a true multi-container deployment first, then "
                "heartbeat and timeout on every RPC, plus leader "
                "re-election for the server role.",
            ],
            [
                "Log-sink disk exhaustion",
                "Log-rotation wrapper around the CSV sinks plus a "
                "pre-run free-space check.",
            ],
            [
                "Checkpoint corruption",
                "Atomic write (write-to-temp then rename), integrity "
                "sidecar (for example SHA-256), and a bounded ring of "
                "historical checkpoints.",
            ],
        ],
        col_widths_cm=[5.5, 11.0],
    )

    add_para(
        doc,
        "Scope note. The set of addressed failure classes reflects a "
        "laptop-scale, single-host execution environment. A multi-host "
        "deployment would change the threat model and add network-level "
        "failures that are explicitly excluded here; migrating the failure "
        "model accompanies the transport migration discussed in Section 6.",
        italic=True,
    )

    add_evaluation(
        doc,
        "Controller thrashing is the failure class most likely to occur "
        "in practice under realistic conditions; it is also the class with "
        "the strongest mitigation (cooldown, decision window, and "
        "exponential-moving-average smoothing on every controller input). "
        "The principal weakness is the implicit assumption that no process "
        "crashes silently — a reasonable assumption for a single-container "
        "scope, but one that must be revisited before any deployment on a "
        "multi-host cluster. The design is explicit about the failures it "
        "does not attempt to handle, which is the level of candour the "
        "C-5 Evaluate outcome requires.",
    )
    add_page_break(doc)


# --------------------------------------------------------------------------- #
#  Section 5: Performance Modelling                                            #
# --------------------------------------------------------------------------- #


def build_section_performance(doc: Document) -> None:
    add_heading(
        doc,
        "5. Performance Modelling (Amdahl, Gustafson, Message Complexity)",
        level=1,
    )

    add_para(
        doc,
        "The performance model combines three analytical tools: Amdahl's "
        "law bounds the strong-scaling speedup, Gustafson's law projects "
        "weak-scaling behaviour, and closed-form message complexity "
        "quantifies the communication cost per mode. The models jointly "
        "predict the qualitative ranking of the three training regimes and "
        "set expectations for the empirical evaluation planned in "
        "Deliverable 3.",
    )

    add_heading(doc, "5.1 Amdahl's law — serial fraction in synchronous mode", level=2)
    add_para(
        doc,
        "Let T(N) denote the wall-clock time per synchronous logical round "
        "across N workers. Decompose it as T(N) = T_compute / N + T_serial, "
        "where T_serial is the server-side averaging plus the broadcast "
        "equivalent that releases workers. The serial fraction on a "
        "representative single-worker round is then s = T_serial / T(1), "
        "and Amdahl's law gives the strong-scaling speedup bound "
        "S(N) = 1 / ( s + (1 − s) / N ).",
    )
    add_para(
        doc,
        "The bound below is a design-time estimate under the stated "
        "assumption and is not a claim of measurement.",
    )
    add_table(
        doc,
        headers=["Workers N", "Speedup upper bound at s = 0.10"],
        rows=[
            ["2", "1.82×"],
            ["4", "3.08×"],
            ["8", "4.71×"],
        ],
        col_widths_cm=[4.0, 7.0],
    )

    add_heading(doc, "5.2 Gustafson's law — weak-scaling projection", level=2)
    add_para(
        doc,
        "If the global batch size grows in proportion to N (so that each "
        "worker keeps a constant per-worker batch), Gustafson's law "
        "predicts S_weak(N) = N − s (N − 1). For s = 0.10 this yields "
        "approximately 1.9× at N = 2, 3.7× at N = 4, and 7.3× at N = 8 — "
        "nearly linear when the serial fraction is small. The design's "
        "baseline configurations keep the global batch constant across "
        "world sizes for fair comparison, which places the system closer "
        "to Amdahl's strong-scaling regime; weak scaling is a planned "
        "sensitivity sweep for the evaluation report.",
    )

    add_heading(doc, "5.3 Message complexity — closed forms per mode", level=2)
    add_table(
        doc,
        headers=[
            "Mode",
            "Messages per round / update",
            "Reference pattern",
            "Implementation choice",
        ],
        rows=[
            [
                "Synchronous (PS-emulated)",
                "N + 1 per round (N worker pushes plus one broadcast-"
                "equivalent)",
                "Classical gather + broadcast: 2N; ring-allreduce: 2(N − 1).",
                "The design uses N + 1: N pushes plus one counter "
                "increment that releases all workers via shared memory.",
            ],
            [
                "Asynchronous",
                "2 per worker update (1 push + 1 pull, amortised)",
                "Classical asynchronous Parameter Server.",
                "Pull is periodic (every K local steps) so the effective "
                "count is 1 + 1/K.",
            ],
            [
                "Hybrid",
                "α · M_sync(N) + (1 − α) · M_async(N)",
                "Weighted mix of the two regimes.",
                "α is the fraction of wall-clock time the controller "
                "spends in synchronous mode.",
            ],
        ],
        col_widths_cm=[3.2, 4.2, 4.6, 4.6],
    )

    add_heading(doc, "5.4 Predicted throughput consistency", level=2)
    add_para(
        doc,
        "The combined model predicts that hybrid-mode throughput will be "
        "bounded above by the asynchronous baseline and below by the "
        "synchronous baseline, with the mixing coefficient α controlling "
        "the position between them. The model also predicts that "
        "mode-transition overhead is not free: the first averaged update "
        "immediately after an asynchronous segment incurs a small lock-"
        "contention cost that the closed form does not charge. The design "
        "mitigates this by coupling transitions to the cooldown period so "
        "that transition overhead is amortised over many rounds. The "
        "quantitative accuracy of this prediction is a planned evaluation "
        "task for Deliverable 3.",
    )

    add_evaluation(
        doc,
        "The three analytical tools together explain the qualitative "
        "ranking the design targets: synchronous is slow under stragglers, "
        "asynchronous is fast but accuracy-sensitive, and hybrid sits "
        "between them in proportion to the controller's sync-time "
        "fraction. The principal limitation is that the closed forms do "
        "not charge for the transient cost of mode flips (lock contention "
        "plus the averaging stall immediately following a switch); the "
        "constant-factor gap between prediction and measurement is "
        "therefore an expected outcome rather than a contradiction, and "
        "is exactly the quantity the evaluation report will measure.",
    )
    add_page_break(doc)


# --------------------------------------------------------------------------- #
#  Section 6: Scalability Assumptions                                          #
# --------------------------------------------------------------------------- #


def build_section_scalability(doc: Document) -> None:
    add_heading(doc, "6. Scalability Assumptions", level=1)

    add_para(
        doc,
        "The scalability story decomposes into four independent axes: "
        "worker count, model size, dataset size, and cluster "
        "heterogeneity. For each axis, the table below states the "
        "design-time assumption, the expected operating limit within the "
        "current emulated deployment, and the condition that would break "
        "the assumption. These are assumptions; measured scalability "
        "curves are reserved for Deliverable 3.",
    )

    add_heading(doc, "6.1 Assumption table", level=2)
    add_table(
        doc,
        headers=[
            "Axis",
            "Assumption",
            "Expected limit",
            "Breaking condition",
        ],
        rows=[
            [
                "Worker count (N)",
                "A shared gradient queue plus a single parameter lock "
                "scale cleanly on one host up to roughly eight workers.",
                "Approximately 8 workers on a four-core laptop; "
                "approximately 32 on a server-class CPU.",
                "Parameter-lock contention dominates, and the gradient "
                "queue becomes a serialisation funnel; a native "
                "distributed transport is required.",
            ],
            [
                "Model size (parameters)",
                "A ResNet-class model (tens of megabytes per gradient "
                "copy) fits comfortably in shared memory for small N.",
                "Roughly one hundred million parameters on a sixteen-"
                "gigabyte host; per-worker gradient copies scale with N.",
                "Total shared memory exceeds available RAM; further "
                "growth requires model sharding or parameter partitioning.",
            ],
            [
                "Dataset size (samples)",
                "Data is streamed from a bind-mounted volume through a "
                "standard data loader; no full load into RAM is required.",
                "Standard benchmarks fit comfortably in memory; "
                "larger corpora stream from disk.",
                "Disk input/output becomes the dominant cost before "
                "compute does on laptop-class hardware.",
            ],
            [
                "Cluster heterogeneity",
                "The controller reliably detects persistent stragglers; "
                "transient spikes are detectable but can over-trigger "
                "under aggressive thresholds.",
                "Up to one straggler per N workers handled without "
                "accuracy collapse.",
                "Multiple simultaneous stragglers hold the controller in "
                "asynchronous mode for long periods; threshold tuning "
                "becomes scenario-specific.",
            ],
        ],
        col_widths_cm=[3.2, 4.5, 4.0, 4.9],
    )

    add_heading(doc, "6.2 Path to a true multi-container deployment", level=2)
    add_para(
        doc,
        "The migration from the single-container emulation to a native "
        "multi-container deployment is planned as a targeted transport "
        "substitution, not a redesign. The controller, the metrics "
        "collector, the aggregator rule, the failure model, and the "
        "consistency contract all remain unchanged.",
    )
    add_numbered(
        doc,
        [
            "Replace the shared gradient queue with native distributed "
            "primitives: All-Reduce for synchronous aggregation and "
            "point-to-point send/receive for asynchronous gradient flow.",
            "Replace the shared sync-round counter with a one-tensor "
            "broadcast of the mode flag from the server, plus a distributed "
            "barrier at the end of each synchronous round.",
            "Promote the single-container Compose definition to N + 1 "
            "services on a bridge network, configuring the process-group "
            "initialisation variables for each service.",
            "Add per-RPC timeouts and a heartbeat channel so that the "
            "failure model can be extended with the network-level classes "
            "currently listed as out of scope.",
        ],
    )

    add_evaluation(
        doc,
        "The emulation-to-multi-container migration path is itemised and "
        "short because the controller and the aggregator are decoupled "
        "from the transport by design; that decoupling is the most "
        "valuable engineering property of the architecture. The weakness "
        "is that the N = 4 and N = 8 projections remain model-driven at "
        "this stage; empirical confirmation is an evaluation-report task. "
        "Overall, the scalability claims are defensible up to roughly "
        "eight workers on a single host and require a transport "
        "substitution beyond that point.",
    )
    add_page_break(doc)


# --------------------------------------------------------------------------- #
#  Section 7: Implementation Plan                                              #
# --------------------------------------------------------------------------- #


def build_section_implementation(doc: Document) -> None:
    add_heading(doc, "7. Implementation Plan", level=1)

    add_para(
        doc,
        "The implementation plan is organised as a phased execution "
        "schedule, followed by a responsibility matrix, a week-indexed "
        "timeline, and a description of the deployment artefact that will "
        "package the system for reproducible evaluation.",
    )

    add_heading(doc, "7.1 Phased execution plan", level=2)
    add_table(
        doc,
        headers=["Phase", "Objective", "Key deliverables"],
        rows=[
            [
                "Phase 1 — Synchronous baseline",
                "Build a correct distributed-training pipeline using "
                "synchronous aggregation; verify throughput and accuracy "
                "for small world sizes.",
                "Training loop, data loaders, model factories, baseline "
                "CSV logs.",
            ],
            [
                "Phase 2 — Asynchronous baseline",
                "Introduce a parameter-server loop that applies gradients "
                "immediately as they arrive; establish a staleness "
                "accounting scheme.",
                "Parameter-server process, staleness tracker, "
                "asynchronous baseline logs.",
            ],
            [
                "Phase 3 — Hybrid switching",
                "Combine both aggregation rules within one server; add "
                "the controller, the metrics collector, and the shared "
                "mode flag.",
                "Hybrid server loop, controller module, switch log schema.",
            ],
            [
                "Phase 4 — Scenario evaluation and threshold tuning",
                "Exercise the hybrid system under homogeneous, persistent-"
                "straggler, and transient-heterogeneity scenarios; sweep "
                "the controller thresholds.",
                "Scenario configurations, sweep driver, aggregated "
                "comparison tables, figures.",
            ],
            [
                "Phase 5 — Final reporting",
                "Consolidate the empirical evidence into a submission-"
                "ready narrative accompanied by per-seed tables, "
                "throughput plots, and a mode-timeline figure.",
                "Final report, presentation deck, regeneration scripts.",
            ],
        ],
        col_widths_cm=[4.0, 6.0, 6.5],
    )

    add_heading(doc, "7.2 Near-term, low-risk refinement", level=2)
    add_table(
        doc,
        headers=["Step", "Action", "Expected outcome"],
        rows=[
            [
                "1",
                "Apply the most promising threshold configuration from the "
                "Phase 4 sweep to the transient-heterogeneity scenario.",
                "A new configuration file derived from the current "
                "scenario template, identical apart from controller "
                "thresholds.",
            ],
            [
                "2",
                "Execute the scenario under multiple random seeds inside "
                "the reproducible container.",
                "A fresh batch of structured logs that can feed the "
                "analysis pipeline without code changes.",
            ],
            [
                "3",
                "Regenerate the comparison tables and figures from the "
                "updated logs.",
                "Refreshed summary tables with the tuned transient-"
                "scenario results folded in.",
            ],
            [
                "4",
                "Evaluate whether the tuned configuration closes the "
                "transient-scenario accuracy gap while preserving the "
                "time-to-convergence advantage.",
                "A documented verdict that either promotes the tuned "
                "thresholds to the default or explains why further work "
                "is needed.",
            ],
        ],
        col_widths_cm=[1.2, 7.6, 7.7],
    )

    add_heading(doc, "7.3 Longer-term work", level=2)
    add_table(
        doc,
        headers=["ID", "Task", "Effort"],
        rows=[
            [
                "L1",
                "Replace the shared gradient queue with native distributed "
                "send/receive primitives.",
                "M",
            ],
            [
                "L2",
                "Replace the shared sync-round counter with a native "
                "one-tensor broadcast plus a distributed barrier.",
                "S",
            ],
            [
                "L3",
                "Promote the single-container orchestration to a multi-"
                "container deployment on a bridge network.",
                "S",
            ],
            [
                "L4",
                "Extend the data-loading layer to a second benchmark "
                "dataset.",
                "S",
            ],
            [
                "L5",
                "Extend the model factory with a second architecture and "
                "re-run the scenario matrix.",
                "M",
            ],
            [
                "L6",
                "Add heartbeat and timeout logic to the server loop to "
                "enable multi-host reliability.",
                "L",
            ],
            [
                "L7",
                "Publish an end-to-end deployment guide for the multi-"
                "container configuration.",
                "S",
            ],
        ],
        col_widths_cm=[1.0, 12.0, 2.5],
    )

    add_heading(doc, "7.4 Responsibility matrix", level=2)
    add_table(
        doc,
        headers=["Responsibility", "Primary owner", "Supporting owners"],
        rows=[
            [
                "System implementation",
                "Muhammad Haseeb Ul Haq",
                "Muhammad Samama Usman",
            ],
            [
                "Testing and experiment execution",
                "Muhammad Samama Usman",
                "Bilal Rana",
            ],
            [
                "Documentation and reporting",
                "Bilal Rana",
                "Muhammad Moiz",
            ],
            [
                "Presentation and demonstration",
                "Muhammad Moiz",
                "Muhammad Haseeb Ul Haq",
            ],
        ],
        col_widths_cm=[5.0, 5.5, 5.5],
    )

    add_heading(doc, "7.5 Week-indexed timeline", level=2)
    add_table(
        doc,
        headers=["Week", "Milestone", "Planned outcome"],
        rows=[
            [
                "W1",
                "Phase 1 — Synchronous baseline",
                "Working distributed-training loop for small world sizes; "
                "baseline throughput and accuracy logs.",
            ],
            [
                "W2",
                "Phase 2 — Asynchronous baseline",
                "Parameter-server loop with staleness accounting; "
                "baseline asynchronous logs.",
            ],
            [
                "W3",
                "Phase 3 — Hybrid pipeline with controller",
                "Dual-mode server loop, controller module, first logged "
                "mode switch under a straggler scenario.",
            ],
            [
                "W4 – W6",
                "Phase 4 — Scenarios, sweep, containerisation",
                "Scenario configurations, tuned thresholds, and a "
                "reproducible container image ready for evaluation runs.",
            ],
            [
                "W7",
                "Phase 5 — Final reporting",
                "Consolidated figures, per-seed tables, and a submission-"
                "ready narrative.",
            ],
            [
                "W8",
                "Deliverable 2 submission (this document)",
                "Full system design document reviewed and submitted.",
            ],
            [
                "W9",
                "Near-term refinement (Section 7.2)",
                "Tuned transient-scenario results folded into the "
                "comparison tables.",
            ],
            [
                "W10",
                "Deliverable 3 — Evaluation and demonstration",
                "Measured speedups, sensitivity sweep, live demonstration.",
            ],
        ],
        col_widths_cm=[2.0, 5.0, 9.5],
    )

    add_heading(doc, "7.6 Deployment artefact", level=2)
    add_para(
        doc,
        "The planned deployment artefact is a versioned container image "
        "that bundles the runtime, the Python dependencies, and the "
        "project code. A companion orchestration file declares the "
        "execution services and bind-mounts the input, output, and "
        "checkpoint directories so that every artefact produced inside "
        "the container is immediately visible on the host. A grader is "
        "expected to be able to rebuild the image, launch a scenario, and "
        "regenerate the analysis tables with a small number of "
        "commands, without installing any host-side Python package.",
    )

    add_evaluation(
        doc,
        "The phased plan is actionable: each phase has a well-defined "
        "deliverable that feeds the next, and the responsibility matrix "
        "distributes the work across the four team members along "
        "complementary axes. The most ambitious items in the longer-term "
        "block (the distributed transport migration, the heartbeat "
        "protocol) are scoped as effort estimates rather than committed "
        "milestones, which is the honest framing at the design stage. The "
        "plan would be strengthened further by a formal acceptance "
        "criterion per phase; this is recommended for the evaluation "
        "report.",
    )
    add_page_break(doc)


# --------------------------------------------------------------------------- #
#  Section 8: Risk Analysis                                                    #
# --------------------------------------------------------------------------- #


def build_section_risks(doc: Document) -> None:
    add_heading(doc, "8. Risk Analysis", level=1)

    add_para(
        doc,
        "The risk register below enumerates the most significant risks to "
        "the design, each scored on a three-level scale for likelihood "
        "and impact. Each entry also identifies the currently planned "
        "mitigation, the proposed stronger mitigation, and the residual "
        "risk after both are in place.",
    )

    add_heading(doc, "8.1 Risk register", level=2)
    add_table(
        doc,
        headers=[
            "ID",
            "Risk",
            "Likelihood",
            "Impact",
            "Level",
            "Current mitigation",
            "Proposed mitigation",
            "Residual",
        ],
        rows=[
            [
                "R1",
                "Controller over-switching on transient stragglers",
                "High",
                "High",
                "[HIGH]",
                "Cooldown period, decision window, and EMA smoothing on "
                "every controller signal.",
                "Scenario-specific thresholds; a second-order signal on "
                "the loss trend; hysteresis tied to validation accuracy.",
                "Medium",
            ],
            [
                "R2",
                "CPU-only hardware ceiling on throughput",
                "High",
                "Medium",
                "[MED]",
                "Short training budgets and relative-throughput reporting.",
                "Execute the scenario matrix on a GPU host when available; "
                "the code path is CPU-only by configuration, not by "
                "design.",
                "Low (for conclusions); high (for absolute numbers).",
            ],
            [
                "R3",
                "Log-vintage heterogeneity across scenarios",
                "Medium",
                "Medium",
                "[MED]",
                "Vintage differences are called out in the evaluation "
                "documentation.",
                "Single clean re-export under a pinned image for the "
                "baselines that predate the final logging schema.",
                "Low",
            ],
            [
                "R4",
                "Container drift from unpinned dependency versions",
                "Medium",
                "High",
                "[HIGH]",
                "Requirements file lists lower bounds for each dependency.",
                "Pin every dependency to an exact version and pin the "
                "base image to an exact tag or digest.",
                "Low once pinned.",
            ],
            [
                "R5",
                "Checkpoint bloat during threshold sweeps",
                "Medium",
                "Low",
                "[LOW]",
                "Only the final checkpoint per run is persisted.",
                "Add a configuration flag to disable checkpointing in "
                "sweep runs and prune historical checkpoints by "
                "configuration.",
                "Very low",
            ],
            [
                "R6",
                "Reproducibility loss if sweep logs are not preserved",
                "Low",
                "High",
                "[MED]",
                "Aggregated sweep summaries are kept alongside the raw "
                "logs.",
                "Commit the canonical merged logs and sweep summary to "
                "version control under large-file tracking.",
                "Low once committed.",
            ],
            [
                "R7",
                "Deadlock at mode transition for larger worker counts",
                "Low",
                "High",
                "[MED]",
                "Transition handler releases workers waiting on the sync-"
                "round counter; covered by smoke tests.",
                "Add an automated multi-worker test that forces a mid-run "
                "flip and asserts all workers exit within a deadline.",
                "Low once tested",
            ],
        ],
        col_widths_cm=[0.9, 3.6, 1.6, 1.3, 1.1, 3.0, 3.0, 1.5],
    )

    add_heading(doc, "8.2 Top three risks", level=2)
    add_para(
        doc,
        "The three highest-priority risks are, in order: (1) controller "
        "over-switching on transient stragglers, because it is the single "
        "risk that directly threatens the accuracy target under one of "
        "the three scenarios and therefore the primary evaluation claim; "
        "(2) container drift from unpinned dependency versions, because "
        "it threatens reproducibility every time the image is rebuilt "
        "from scratch; and (3) reproducibility loss if sweep logs are "
        "displaced, because the threshold sweep is a first-class part of "
        "the argument that the design can close its own known gaps. The "
        "first risk is mitigated by further threshold tuning, the second "
        "by exact version pinning, and the third by storing the merged "
        "logs under version control.",
    )

    add_evaluation(
        doc,
        "Every high-severity risk in the register has a concrete "
        "mitigation that is either a small configuration edit or a "
        "single-command operation. The honest weakness is that the "
        "highest-priority risk — controller over-switching — is a "
        "policy-design problem rather than a process problem: it can be "
        "made less likely with better thresholds but is unlikely to "
        "disappear entirely until a richer signal (for example a two-"
        "order-of-magnitude observation window) is added to the "
        "controller input. Naming this gap explicitly is itself a design "
        "contribution at the C-5 level.",
    )
    add_page_break(doc)


# --------------------------------------------------------------------------- #
#  Conclusion                                                                  #
# --------------------------------------------------------------------------- #


def build_conclusion(doc: Document) -> None:
    add_heading(doc, "9. Conclusion and C-5 Summary", level=1)

    add_para(
        doc,
        "This design document defines a hybrid distributed SGD system "
        "whose distinguishing property is a bi-directional, online "
        "controller that switches between synchronous and asynchronous "
        "aggregation based on live signals. The sections above have laid "
        "out the architecture (Section 1), the communication primitives "
        "and their transport (Section 2), the per-mode consistency "
        "contract (Section 3), the failure classes the design does and "
        "does not address (Section 4), closed-form performance bounds "
        "from Amdahl's law, Gustafson's law, and message complexity "
        "(Section 5), scalability assumptions along four axes (Section "
        "6), a phased execution plan with responsibility and timeline "
        "(Section 7), and a risk register with mitigations (Section 8).",
    )

    add_heading(doc, "9.1 Mapping to the C-5 (Evaluate) learning outcome", level=2)
    add_table(
        doc,
        headers=["C-5 judgement dimension", "Where it is demonstrated"],
        rows=[
            [
                "Model definition",
                "Communication primitives (§2), consistency classes (§3), "
                "failure classes (§4), Amdahl/Gustafson/message complexity "
                "(§5), scalability axes (§6).",
            ],
            [
                "Choice justification",
                "Every model section ends with why the chosen approach "
                "fits the problem (e.g. §2 transport rationale, §3 "
                "consistency-switch semantics).",
            ],
            [
                "Trade-off analysis",
                "Explicit trade-off columns in the communication tables, "
                "the consistency matrix, the failure residuals, and the "
                "risk register.",
            ],
            [
                "Design judgement",
                "An Evaluation paragraph closes every section, stating the "
                "strength, the weakness, and what C-5 actually means for "
                "that section.",
            ],
        ],
        col_widths_cm=[5.5, 11.0],
    )

    add_para(
        doc,
        "Closing note. The document is intentionally forward-looking. "
        "It specifies what the system is designed to achieve, what it is "
        "designed to protect against, and how it is designed to behave "
        "as the workload and hardware change. The measurements that will "
        "verify — or refine — each of these predictions constitute the "
        "scope of Deliverable 3.",
    )


# --------------------------------------------------------------------------- #
#  Top-level builder                                                           #
# --------------------------------------------------------------------------- #


def build_document() -> Document:
    doc = Document()
    configure_styles(doc)
    build_cover(doc)
    build_toc(doc)
    build_executive_summary(doc)
    build_section_architecture(doc)
    build_section_communication(doc)
    build_section_consistency(doc)
    build_section_failure(doc)
    build_section_performance(doc)
    build_section_scalability(doc)
    build_section_implementation(doc)
    build_section_risks(doc)
    build_conclusion(doc)
    return doc


def main() -> None:
    DOCX_OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = build_document()
    doc.save(DOCX_OUT)
    try:
        rel = DOCX_OUT.relative_to(REPO_ROOT)
    except ValueError:
        rel = DOCX_OUT
    print(f"[build_design_document] Wrote: {rel}")


if __name__ == "__main__":
    main()
