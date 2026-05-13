"""
Build the demo presentation PPT from diagrams and proof images.
Run:  python demo_recording/build_ppt.py
"""
import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

BASE = os.path.dirname(os.path.abspath(__file__))
DIAGRAMS = os.path.join(BASE, "diagrams")
PROOF    = os.path.join(BASE, "proof")
OUT      = os.path.join(BASE, "Hybrid_Distributed_SGD_Demo.pptx")

prs = Presentation()
prs.slide_width  = Inches(13.333)
prs.slide_height = Inches(7.5)

# ── Colour palette ──────────────────────────────────────────────────────
BG_DARK   = RGBColor(0x0F, 0x17, 0x2A)
ACCENT    = RGBColor(0x00, 0xB4, 0xD8)
WHITE     = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GRAY = RGBColor(0xCC, 0xCC, 0xCC)
GOLD      = RGBColor(0xFF, 0xD6, 0x00)

def _set_bg(slide, color=BG_DARK):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color

def _add_title_slide(title_text, subtitle_text=""):
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    _set_bg(slide)
    # Title
    txBox = slide.shapes.add_textbox(Inches(0.8), Inches(2.0), Inches(11.7), Inches(2.0))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title_text
    p.font.size = Pt(44)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.alignment = PP_ALIGN.CENTER
    if subtitle_text:
        p2 = tf.add_paragraph()
        p2.text = subtitle_text
        p2.font.size = Pt(22)
        p2.font.color.rgb = ACCENT
        p2.alignment = PP_ALIGN.CENTER
    return slide

def _add_image_slide(title_text, image_path, subtitle_text=""):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_bg(slide)
    # Title bar
    txBox = slide.shapes.add_textbox(Inches(0.5), Inches(0.2), Inches(12.3), Inches(0.9))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title_text
    p.font.size = Pt(30)
    p.font.bold = True
    p.font.color.rgb = ACCENT
    p.alignment = PP_ALIGN.LEFT
    if subtitle_text:
        p2 = tf.add_paragraph()
        p2.text = subtitle_text
        p2.font.size = Pt(16)
        p2.font.color.rgb = LIGHT_GRAY
        p2.alignment = PP_ALIGN.LEFT
    # Image – centered, max size
    if os.path.exists(image_path):
        # Calculate image size to fit within bounds
        max_w = Inches(11.5)
        max_h = Inches(5.8)
        from PIL import Image as PILImage
        with PILImage.open(image_path) as img:
            iw, ih = img.size
        ratio = min(max_w / Emu(iw * 914400 // 96), max_h / Emu(ih * 914400 // 96))
        w = int(iw * 914400 // 96 * ratio)
        h = int(ih * 914400 // 96 * ratio)
        left = int((prs.slide_width - w) / 2)
        top = Inches(1.3)
        slide.shapes.add_picture(image_path, left, top, w, h)
    return slide

def _add_text_slide(title_text, bullet_lines):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_bg(slide)
    # Title
    txBox = slide.shapes.add_textbox(Inches(0.5), Inches(0.2), Inches(12.3), Inches(0.9))
    tf = txBox.text_frame
    p = tf.paragraphs[0]
    p.text = title_text
    p.font.size = Pt(32)
    p.font.bold = True
    p.font.color.rgb = ACCENT
    # Body
    txBox2 = slide.shapes.add_textbox(Inches(0.8), Inches(1.3), Inches(11.7), Inches(5.5))
    tf2 = txBox2.text_frame
    tf2.word_wrap = True
    for i, line in enumerate(bullet_lines):
        if i == 0:
            p = tf2.paragraphs[0]
        else:
            p = tf2.add_paragraph()
        p.text = line
        p.font.size = Pt(20)
        p.font.color.rgb = WHITE
        p.space_after = Pt(10)
    return slide

def _add_table_slide(title_text, headers, rows):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_bg(slide)
    txBox = slide.shapes.add_textbox(Inches(0.5), Inches(0.15), Inches(12.3), Inches(0.7))
    tf = txBox.text_frame
    p = tf.paragraphs[0]
    p.text = title_text
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = ACCENT

    n_rows = len(rows) + 1
    n_cols = len(headers)
    tbl_w = Inches(12.0)
    tbl_h = Inches(5.5)
    left = Inches(0.65)
    top = Inches(1.1)
    table_shape = slide.shapes.add_table(n_rows, n_cols, left, top, tbl_w, tbl_h)
    table = table_shape.table

    # Header row
    for ci, h in enumerate(headers):
        cell = table.cell(0, ci)
        cell.text = h
        for para in cell.text_frame.paragraphs:
            para.font.size = Pt(14)
            para.font.bold = True
            para.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
            para.alignment = PP_ALIGN.CENTER
        from pptx.oxml.ns import qn
        tc = cell._tc
        tcPr = tc.get_or_add_tcPr()
        solidFill = tcPr.makeelement(qn('a:solidFill'), {})
        srgbClr = solidFill.makeelement(qn('a:srgbClr'), {'val': '00B4D8'})
        solidFill.append(srgbClr)
        tcPr.append(solidFill)

    # Data rows
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = table.cell(ri + 1, ci)
            cell.text = str(val)
            for para in cell.text_frame.paragraphs:
                para.font.size = Pt(13)
                para.font.color.rgb = WHITE
                para.alignment = PP_ALIGN.CENTER
            from pptx.oxml.ns import qn
            tc = cell._tc
            tcPr = tc.get_or_add_tcPr()
            solidFill = tcPr.makeelement(qn('a:solidFill'), {})
            bg_hex = '1A2744' if ri % 2 == 0 else '0F172A'
            srgbClr = solidFill.makeelement(qn('a:srgbClr'), {'val': bg_hex})
            solidFill.append(srgbClr)
            tcPr.append(solidFill)
    return slide


# ═══════════════════════════════════════════════════════════════════════
#  BUILD SLIDES
# ═══════════════════════════════════════════════════════════════════════

# 1. Title
_add_title_slide(
    "Hybrid Distributed SGD",
    "Bi-directional Switching: Ring All-Reduce ↔ Parameter Server\n"
    "Parallel and Distributed Computing — Spring 2026, SEECS NUST"
)

# 2. Problem Statement
_add_image_slide(
    "1 — Why Distributed Training?",
    os.path.join(DIAGRAMS, "01_problem_statement.png"),
    "Split work across multiple workers to reduce training time"
)

# 3. Sync vs Async
_add_image_slide(
    "2 — Two Standard Approaches: Sync vs Async",
    os.path.join(DIAGRAMS, "02_sync_vs_async.png"),
    "Sync is accurate but slow (stragglers). Async is fast but stale."
)

# 4. The Hybrid Idea
_add_image_slide(
    "3 — Our Solution: Hybrid Bi-directional Switching",
    os.path.join(DIAGRAMS, "03_hybrid_switching.png"),
    "Switch Sync↔Async at runtime based on live signals"
)

# 5. Controller Signals
_add_image_slide(
    "4 — Controller Signals & Thresholds",
    os.path.join(DIAGRAMS, "08_controller_signals.png"),
    "Worker Time Variance • Gradient Norm Variance • Loss Trend (all EMA-smoothed)"
)

# 6. Architecture
_add_image_slide(
    "5 — System Architecture",
    os.path.join(DIAGRAMS, "04_system_architecture.png"),
    "1 PS + 2 Workers inside Docker — shared memory + mp.Queue"
)

# 7. Setup text slide
_add_text_slide("6 — Experimental Setup", [
    "• Dataset: CIFAR-10 — 60,000 colour images (32×32), 10 classes",
    "• Model: ResNet-18 (CIFAR-adapted) — 11.17 M parameters (~44 MB gradients)",
    "• Workers: 2  |  Batch size: 32  |  Optimizer: SGD (momentum 0.9, lr 0.01)",
    "• Random seeds: 42, 123, 777  (3 per condition for reproducibility)",
    "• Containerized: Docker image pdc-hybrid:latest  |  CPU-only, 4 cores, 6 GB RAM",
    "• Total runs: 24 main + 17 threshold sweep = 41 experiments",
    "",
    "Target metric: Time-to-Accuracy (TTA) at 40% validation accuracy",
])

# 8. Pipeline
_add_image_slide(
    "7 — Project Pipeline (5 Phases)",
    os.path.join(DIAGRAMS, "07_project_pipeline.png"),
    "Phase 1: Sync → Phase 2: Async → Phase 3: Hybrid → Phase 4: Experiments → Phase 5: Analysis"
)

# 9. Three Scenarios
_add_image_slide(
    "8 — Three Test Scenarios",
    os.path.join(DIAGRAMS, "05_three_scenarios.png"),
    "S0: Homogeneous  |  SA: Persistent straggler  |  SB: Transient straggler"
)

# 10. Metrics slide
_add_text_slide("9 — Evaluation Metrics (7 metrics)", [
    "1. Time-to-Accuracy (TTA)  — Seconds to reach 40% val accuracy",
    "2. Final Accuracy (%)  — Validation accuracy at end of training",
    "3. Wall Clock Time (s)  — Total experiment runtime",
    "4. Throughput (img/s)  — Images processed per second",
    "5. Staleness (steps)  — How outdated a worker's model was",
    "6. Switch Count  — Number of Sync↔Async mode flips",
    "7. Sync %  — Fraction of time spent in synchronous mode",
])

# 11. Final Results diagram
_add_image_slide(
    "10 — Scenario A Summary: Hybrid Wins",
    os.path.join(DIAGRAMS, "06_final_results.png"),
    "Hybrid: 12% faster than Sync, nearly double throughput, similar accuracy"
)

# 12. TTA Bars (proof)
_add_image_slide(
    "11 — Time-to-Accuracy Comparison",
    os.path.join(PROOF, "phase5_tta_bars.png"),
    "How fast each mode reaches 40% accuracy — numbers above bars show seeds that reached target"
)

# 13. Hybrid Mode Timeline (proof)
_add_image_slide(
    "12 — Hybrid Mode Timeline",
    os.path.join(PROOF, "phase5_hybrid_mode_timeline.png"),
    "Blue = Sync, Orange = Async — vertical lines are switch events with trigger reasons"
)

# 14. Key Findings
_add_text_slide("16 — Key Findings", [
    "✓  Bi-directional switching is REAL and LOGGED — both sync→async and async→sync",
    "✓  Scenario A: Hybrid is 12% faster than Sync (1233s vs 1409s TTA)",
    "✓  Scenario A: Hybrid accuracy (41.9%) within variance of Sync (43.2%)",
    "✓  Scenario A: Hybrid throughput nearly 2× Sync (23.3 vs 13.6 img/s)",
    "✓  Scenario A: Hybrid staleness (359) lower than pure Async (373)",
    "",
    "△  Scenario B: Hybrid fastest (864s) but accuracy drops (36.9%) — controller over-reacts",
    "△  Threshold sweep (17 configs) identifies tuning path to fix Scenario B",
    "",
    "⚑  40% target is a BENCHMARK LINE, not the project goal —",
    "    the goal is proving adaptive switching saves time, not maximizing accuracy",
])

# 18. Conclusion
_add_title_slide(
    "Thank You",
    "Questions?"
)

# ═══════════════════════════════════════════════════════════════════════
prs.save(OUT)
print(f"OK - Presentation saved to: {OUT}")
print(f"   Slides: {len(prs.slides)}")
