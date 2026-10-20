import io
import hashlib
import sqlite3
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Pricing Analytics Decision Lab",
    page_icon="📈",
    layout="wide",
)

# -----------------------------
# Core models and optimization
# -----------------------------

DAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

BASELINE = pd.read_csv("data/hotel_parameters.csv")

# Normalize CSV headers so accidental spaces/tabs do not cause KeyError.
BASELINE.columns = BASELINE.columns.str.strip()

REQUIRED_BASELINE_COLUMNS = {"Day", "D", "a", "b"}
missing_columns = REQUIRED_BASELINE_COLUMNS - set(BASELINE.columns)
if missing_columns:
    st.error(
        "data/hotel_parameters.csv is missing required columns: "
        + ", ".join(sorted(missing_columns))
        + ". Expected columns are: Day, D, a, b."
    )
    st.stop()

# Normalize day labels and validate the expected seven-day schedule.
BASELINE["Day"] = BASELINE["Day"].astype(str).str.strip()

BASE_CAPACITY = 120.0
BASE_COST = 0.0
PRICE_MIN = 0.0
PRICE_MAX = 300.0
PRICE_STEP = 0.01


def logit_demand(price, D, a, b):
    z = np.clip(a + b * price, -700, 700)
    return D * np.exp(z) / (1.0 + np.exp(z))


def served_demand(price, D, a, b, capacity):
    return min(float(logit_demand(price, D, a, b)), float(capacity))


def revenue_at(price, D, a, b, capacity):
    return float(price) * served_demand(price, D, a, b, capacity)


def profit_at(price, D, a, b, capacity, cost=0.0):
    q = served_demand(price, D, a, b, capacity)
    return (float(price) - float(cost)) * q


def optimize_day(D, a, b, capacity=BASE_CAPACITY, cost=BASE_COST):
    prices = np.arange(PRICE_MIN, PRICE_MAX + PRICE_STEP, PRICE_STEP)
    profits = np.array([profit_at(p, D, a, b, capacity, cost) for p in prices])
    idx = int(np.argmax(profits))
    p = float(prices[idx])
    q = served_demand(p, D, a, b, capacity)
    raw = float(logit_demand(p, D, a, b))
    return {
        "price": p,
        "demand": raw,
        "served": float(q),
        "capacity_utilization": float(q / capacity) if capacity else 0.0,
        "revenue": float(p * q),
        "profit": float((p - cost) * q),
    }


def evaluate_schedule(prices_by_day, data=BASELINE, capacity=BASE_CAPACITY, cost=BASE_COST):
    rows = []
    for _, r in data.iterrows():
        day = r["Day"]
        price = float(prices_by_day[day])
        raw = float(logit_demand(price, r["D"], r["a"], r["b"]))
        served = min(raw, capacity)
        rows.append(
            {
                "Day": day,
                "Price": price,
                "Expected Demand": raw,
                "Rooms Sold": served,
                "Capacity Utilization": served / capacity if capacity else 0.0,
                "Revenue": price * served,
                "Profit": (price - cost) * served,
                "Unserved Demand": max(raw - capacity, 0.0),
            }
        )
    return pd.DataFrame(rows)


def variable_optimum(data=BASELINE, capacity=BASE_CAPACITY, cost=BASE_COST):
    rows = []
    for _, r in data.iterrows():
        o = optimize_day(r["D"], r["a"], r["b"], capacity, cost)
        rows.append(
            {
                "Day": r["Day"],
                "Price": o["price"],
                "Expected Demand": o["demand"],
                "Rooms Sold": o["served"],
                "Capacity Utilization": o["capacity_utilization"],
                "Revenue": o["revenue"],
                "Profit": o["profit"],
                "Unserved Demand": max(o["demand"] - capacity, 0.0),
            }
        )
    return pd.DataFrame(rows)


def money(x):
    return f"${x:,.0f}"


def money2(x):
    return f"${x:,.2f}"


def render_styled_table(df, formats=None, first_col_left=True):
    """Render a consistent bordered HTML table for the student-facing lab."""
    formats = formats or {}
    display_df = df.copy()

    for col, fmt in formats.items():
        if col in display_df.columns:
            display_df[col] = display_df[col].map(lambda x: fmt.format(x) if pd.notna(x) else "")

    header_html = "".join(f"<th>{col}</th>" for col in display_df.columns)
    rows_html = []
    for _, row in display_df.iterrows():
        cells = []
        for i, value in enumerate(row):
            cls = ' class="first-col"' if first_col_left and i == 0 else ""
            cells.append(f"<td{cls}>{value}</td>")
        rows_html.append("<tr>" + "".join(cells) + "</tr>")

    st.markdown(
        """
        <style>
        .pricing-table-wrap { overflow-x: auto; margin: 10px 0 18px 0; }
        .pricing-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 15px;
            background: #ffffff;
        }
        .pricing-table th {
            background-color: #003b5c;
            color: #ffffff;
            padding: 10px 12px;
            border: 1px solid #cfd8e3;
            text-align: center;
            font-weight: 700;
            white-space: nowrap;
        }
        .pricing-table td {
            padding: 9px 12px;
            border: 1px solid #cfd8e3;
            text-align: center;
            background-color: #ffffff;
        }
        .pricing-table tr:nth-child(even) td { background-color: #f7fafc; }
        .pricing-table td.first-col { text-align: left; font-weight: 600; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    table_html = (
        '<div class="pricing-table-wrap"><table class="pricing-table">'
        f'<thead><tr>{header_html}</tr></thead>'
        f'<tbody>{"".join(rows_html)}</tbody></table></div>'
    )
    st.markdown(table_html, unsafe_allow_html=True)


def money_md(x, decimals=0):
    """Format currency safely inside Streamlit Markdown/alert text.

    Streamlit treats an unescaped dollar sign as a LaTeX delimiter.
    Escaping the dollar sign prevents currency values from being rendered
    as math and breaking surrounding Markdown bold formatting.
    """
    if decimals == 2:
        return f"\\${x:,.2f}"
    return f"\\${x:,.0f}"


def reset_lab():
    for key in list(st.session_state.keys()):
        if key.startswith("lab_") or key.startswith("price_") or key.startswith("shock_") or key.startswith("seg_"):
            del st.session_state[key]


# -----------------------------
# PDF report generation
# -----------------------------

EASTERN_TZ = ZoneInfo("America/New_York")
REVEAL_UNLOCK = datetime(2026, 11, 5, 0, 0, 0, tzinfo=EASTERN_TZ)
MAX_ATTEMPTS = 3
ATTEMPT_DB = Path("/tmp/ban517_pricing_attempts.sqlite3")

def reveal_is_unlocked():
    return datetime.now(EASTERN_TZ) >= REVEAL_UNLOCK

def _student_key(student_id):
    return hashlib.sha256(student_id.strip().lower().encode("utf-8")).hexdigest()

def _attempt_conn():
    conn = sqlite3.connect(ATTEMPT_DB)
    conn.execute("CREATE TABLE IF NOT EXISTS attempts (student_key TEXT PRIMARY KEY, completed INTEGER NOT NULL DEFAULT 0)")
    conn.commit()
    return conn

def completed_attempts(student_id):
    if not student_id or not student_id.strip(): return 0
    conn = _attempt_conn()
    try:
        row = conn.execute("SELECT completed FROM attempts WHERE student_key=?", (_student_key(student_id),)).fetchone()
        return int(row[0]) if row else 0
    finally:
        conn.close()

def record_completed_attempt(student_id):
    conn = _attempt_conn()
    try:
        key = _student_key(student_id)
        row = conn.execute("SELECT completed FROM attempts WHERE student_key=?", (key,)).fetchone()
        current = int(row[0]) if row else 0
        if current >= MAX_ATTEMPTS: return current
        new_count = current + 1
        conn.execute("INSERT INTO attempts(student_key,completed) VALUES(?,?) ON CONFLICT(student_key) DO UPDATE SET completed=excluded.completed", (key,new_count))
        conn.commit()
        return new_count
    finally:
        conn.close()


def make_verification_code(student_id, generated_at):
    """Create a short deterministic fingerprint from student ID + exact timestamp."""
    payload = f"{student_id.strip()}|{generated_at.isoformat(timespec='microseconds')}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12].upper()


def build_pdf_report(student_name, student_id, course_section, generated_at, verification_code,
                     manual_locked, opt_df, cap, dem, seg, predictions, benchmarks_visible, attempt_number):
    """Build the student's pricing lab report entirely in memory."""
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.58 * inch,
        bottomMargin=0.72 * inch,
        title="BAN 517 Pricing Analytics Decision Lab Results",
        author="BAN 517 Supply Chain Analytics",
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="LabTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#003b5c"),
        alignment=TA_CENTER,
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="LabSubTitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#006c67"),
        alignment=TA_CENTER,
        spaceAfter=12,
    ))
    styles.add(ParagraphStyle(
        name="Section",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#003b5c"),
        spaceBefore=10,
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="BodySmall",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="Verify",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#003b5c"),
        alignment=TA_LEFT,
        borderColor=colors.HexColor("#008c95"),
        borderWidth=0.7,
        borderPadding=6,
        backColor=colors.HexColor("#f4fbfb"),
        spaceBefore=8,
        spaceAfter=8,
    ))

    def fmt_money(x):
        return f"${float(x):,.2f}"

    def fmt_num(x):
        return f"{float(x):,.0f}"

    def styled_table(data, col_widths=None, font_size=8):
        t = Table(data, colWidths=col_widths, repeatRows=1, hAlign="CENTER")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#003b5c")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), font_size),
            ("LEADING", (0, 0), (-1, -1), font_size + 2),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cfd8df")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f9fb")]),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ]))
        return t

    generated_eastern = generated_at.astimezone(EASTERN_TZ)
    generated_utc = generated_at.astimezone(ZoneInfo("UTC"))
    time_display = generated_eastern.strftime("%Y-%m-%d %H:%M:%S.%f %Z")
    utc_display = generated_utc.strftime("%Y-%m-%d %H:%M:%S.%f UTC")

    story = []
    story.append(Paragraph("Pricing Analytics Decision Lab", styles["LabTitle"]))
    story.append(Paragraph("BAN 517 - Supply Chain Analytics | Student Results Report", styles["LabSubTitle"]))

    student_table = [
        ["Student", student_name or "Not entered", "Student ID", student_id],
        ["Section", course_section or "-", "Attempt", f"{attempt_number} of {MAX_ATTEMPTS}"],
        ["Generated", time_display, "Benchmark status", "Visible" if benchmarks_visible else "Locked until Nov. 5, 2026"],
    ]
    stbl = Table(student_table, colWidths=[0.8*inch, 2.05*inch, 0.8*inch, 3.0*inch])
    stbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#eef7ff")),
        ("BACKGROUND", (2,0), (2,-1), colors.HexColor("#eef7ff")),
        ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME", (2,0), (2,-1), "Helvetica-Bold"),
        ("FONTNAME", (1,0), (1,-1), "Helvetica"),
        ("FONTNAME", (3,0), (3,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 8.5),
        ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#cfd8df")),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    story.append(stbl)
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        f"<b>Verification code:</b> {verification_code}<br/>"
        f"<b>Exact generation time:</b> {time_display}<br/>"
        f"<b>UTC equivalent:</b> {utc_display}",
        styles["Verify"],
    ))

    story.append(Paragraph("Stage 1 - Weekly Hotel Pricing Prediction and Strategy", styles["Section"]))
    story.append(Paragraph(
        f"Pre-analysis prediction - three days expected to support the highest prices: <b>{predictions['stage1_high_days']}</b>",
        styles["BodySmall"],
    ))
    manual_rows = [["Day", "Price", "Expected Demand", "Rooms Sold", "Capacity Used", "Revenue"]]
    for _, r in manual_locked.iterrows():
        manual_rows.append([
            r["Day"], fmt_money(r["Price"]), fmt_num(r["Expected Demand"]),
            fmt_num(r["Rooms Sold"]), f"{100*r['Capacity Utilization']:.1f}%", fmt_money(r["Revenue"])
        ])
    story.append(styled_table(manual_rows, [0.9*inch, 0.8*inch, 1.15*inch, 1.0*inch, 1.05*inch, 1.0*inch]))
    story.append(Spacer(1, 5))
    story.append(Paragraph(
        f"Student weekly room revenue: <b>{fmt_money(manual_locked['Revenue'].sum())}</b> | "
        f"Weekly occupied room-nights: <b>{fmt_num(manual_locked['Rooms Sold'].sum())}</b>",
        styles["BodySmall"],
    ))

    story.append(Paragraph("Stage 2 - Optimization Benchmark", styles["Section"]))
    if benchmarks_visible:
        benchmark_rows = [["Day", "Student Price", "Optimized Price", "Student Revenue", "Optimized Revenue"]]
        for i, day in enumerate(DAYS):
            benchmark_rows.append([day, fmt_money(manual_locked.iloc[i]["Price"]), fmt_money(opt_df.iloc[i]["Price"]), fmt_money(manual_locked.iloc[i]["Revenue"]), fmt_money(opt_df.iloc[i]["Revenue"])])
        story.append(styled_table(benchmark_rows, [1.0*inch,1.05*inch,1.05*inch,1.2*inch,1.2*inch]))
    else:
        story.append(Paragraph("Optimization benchmarks are intentionally withheld until November 5, 2026.", styles["BodySmall"]))

    story.append(PageBreak())
    story.append(Paragraph("Stage 3 - Hotel Capacity Shock", styles["Section"]))
    cap_rows = [
        ["Pre-analysis prediction", "Student Response"],
        ["Expected direction of optimal price", predictions["stage3_direction"]],
        ["Predicted optimal Saturday room rate", fmt_money(predictions["stage3_predicted_price"])],
    ]
    story.append(styled_table(cap_rows, [2.8*inch, 2.6*inch]))
    story.append(Spacer(1, 5))
    cap_result_rows = [
        ["Scenario", "Student Price", "Expected Demand", "Rooms Sold", "Revenue", "Optimized Price"],
        ["Saturday capacity = 90", fmt_money(cap["student_price"]), fmt_num(cap["expected_demand"]),
         fmt_num(cap["rooms_sold"]), fmt_money(cap["revenue"]), fmt_money(cap.get("optimized_price", 0)) if benchmarks_visible else "Locked"],
    ]
    story.append(styled_table(cap_result_rows, [1.55*inch, 0.95*inch, 1.0*inch, 0.9*inch, 0.95*inch, 1.0*inch], font_size=7.5))

    story.append(Paragraph("Stage 4 - Hotel Demand Shock", styles["Section"]))
    dem_rows = [
        ["Pre-analysis prediction", "Student Response"],
        ["Expected direction of optimal price", predictions["stage4_direction"]],
        ["Predicted optimal Tuesday room rate", fmt_money(predictions["stage4_predicted_price"])],
    ]
    story.append(styled_table(dem_rows, [2.8*inch, 2.6*inch]))
    story.append(Spacer(1, 5))
    dem_result_rows = [
        ["Scenario", "Student Price", "Expected Demand", "Rooms Sold", "Revenue", "Optimized Price"],
        ["Tuesday D = 230", fmt_money(dem["student_price"]), fmt_num(dem["expected_demand"]),
         fmt_num(dem["rooms_sold"]), fmt_money(dem["revenue"]), fmt_money(dem.get("optimized_price", 0)) if benchmarks_visible else "Locked"],
    ]
    story.append(styled_table(dem_result_rows, [1.55*inch, 0.95*inch, 1.0*inch, 0.9*inch, 0.95*inch, 1.0*inch], font_size=7.5))

    story.append(Paragraph("Stage 5 - Segmented Pricing", styles["Section"]))
    seg_pred_rows = [
        ["Pre-analysis prediction", "Student Response"],
        ["More price-sensitive segment", predictions["stage5_sensitive_segment"]],
        ["Expected price relationship", predictions["stage5_price_relationship"]],
    ]
    story.append(styled_table(seg_pred_rows, [2.8*inch, 2.6*inch]))
    story.append(Spacer(1, 5))
    seg_rows = [["Strategy", "Business Price", "Leisure Price", "Total Demand", "Revenue"]]
    seg_rows.append(["Student segmented prices", fmt_money(seg["segment_A_price"]), fmt_money(seg["segment_B_price"]), f"{seg['total_segment_demand']:.2f}", fmt_money(seg["segmented_revenue"])])
    if benchmarks_visible and "optimized_A_price" in seg:
        seg_rows.append(["Optimized segmented prices", fmt_money(seg["optimized_A_price"]), fmt_money(seg["optimized_B_price"]), f"{seg['optimized_total_demand']:.2f}", fmt_money(seg["optimized_revenue"])])
    story.append(styled_table(seg_rows, [1.8*inch,1.2*inch,1.2*inch,1.1*inch,1.2*inch]))

    story.append(Spacer(1, 10))
    story.append(Paragraph("Completion Record", styles["Section"]))
    story.append(Paragraph(
        "This report documents the student's submitted pricing decisions, pre-analysis predictions, "
        "and the optimization benchmarks revealed after each decision was locked. Conceptual reflection "
        "prompts in the lab are learning checks and are not automatically graded.",
        styles["BodySmall"],
    ))

    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "Submission verification", styles["Section"]
    ))
    story.append(Paragraph(
        f"This report was generated by the Pricing Analytics Decision Lab for <b>{student_name or 'Not entered'}</b> "
        f"(Student ID: <b>{student_id}</b>) at <b>{time_display}</b>. "
        f"Verification code: <b>{verification_code}</b>.",
        styles["BodySmall"],
    ))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#dfe5ec"))
        canvas.setLineWidth(0.5)
        canvas.line(doc.leftMargin, 0.49*inch, letter[0]-doc.rightMargin, 0.49*inch)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#52606d"))
        canvas.drawString(doc.leftMargin, 0.30*inch, f"Generated: {time_display}")
        footer_text = f"Verification code: {verification_code}"
        canvas.drawRightString(letter[0]-doc.rightMargin, 0.30*inch, footer_text)
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


# -----------------------------
# Session state
# -----------------------------

def initialize_state():
    defaults = {
        "lab_stage1_prediction_locked": False,
        "lab_stage1_prediction": None,
        "lab_stage1_locked": False,
        "lab_stage2_revealed": False,
        "lab_stage3_prediction_locked": False,
        "lab_stage3_direction": None,
        "lab_stage3_predicted_price": None,
        "lab_stage3_locked": False,
        "lab_stage3_revealed": False,
        "lab_stage4_prediction_locked": False,
        "lab_stage4_direction": None,
        "lab_stage4_predicted_price": None,
        "lab_stage4_locked": False,
        "lab_stage4_revealed": False,
        "lab_stage5_prediction_locked": False,
        "lab_stage5_sensitive_segment": None,
        "lab_stage5_price_relationship": None,
        "lab_stage5_locked": False,
        "lab_stage5_revealed": False,
        "lab_manual_schedule": None,
        "lab_capacity_shock": None,
        "lab_demand_shock": None,
        "lab_segment_result": None,
    "lab_pdf_bytes": None,
    "lab_pdf_timestamp": None,
    "lab_verification_code": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


initialize_state()

# -----------------------------
# Header
# -----------------------------

st.markdown(
    """
    <div style="background: linear-gradient(135deg,#003b5c 0%,#006c67 55%,#00a6a6 100%);
                padding: 26px 28px; border-radius: 16px; color: white; margin-bottom: 18px;">
      <div style="font-size: 13px; letter-spacing: 1px; text-transform: uppercase; font-weight: 700;">
        BAN 517 · Supply Chain Analytics
      </div>
      <div style="font-size: 32px; line-height: 1.2; font-weight: 800; margin-top: 4px;">
        Pricing Analytics Decision Lab
      </div>
      <div style="font-size: 17px; margin-top: 8px;">
        Solve each pricing problem independently, enter and test your decisions, then benchmark them against optimization.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "**How to use this lab:** Solve the pricing problems independently before finalizing your answers. "
    "Excel Solver is recommended. Use the sliders to enter and test your prices, refine them if needed, "
    "and then lock your decision. Optimization benchmarks and explanatory answers are locked until **November 5, 2026**."
)

if reveal_is_unlocked():
    st.success("Optimization benchmarks are now available for review.")
else:
    st.warning("Optimization benchmarks and Think About It explanations will unlock on November 5, 2026.")

with st.expander("Recommended Excel Solver setup", expanded=False):
    st.markdown(
        r"""
For the weekly hotel-pricing model, use two decision variables for each day:

- **Price (p):** room rate.
- **Rooms Sold (x):** number of rooms actually sold.

The demand model is

$$Q(p)=D\frac{e^{a+bp}}{1+e^{a+bp}}$$

and the objective is

$$\max \sum_i p_i x_i.$$

Recommended Solver setup:

- **Set Objective:** Total Revenue; choose **Max**.
- **Changing Variable Cells:** all Price $(p)$ and Rooms Sold $(x)$ cells.
- **Demand constraint:** $x_i \leq Q_i(p_i)$.
- **Capacity constraint:** $x_i \leq 120$ for the baseline problem.
- **Price bound:** $0 \leq p_i \leq 300$.
- **Solving Method:** **GRG Nonlinear**.
- Use positive starting values (for example, $p=150$ and $x=50$) rather than starting both decision variables at zero.

For the segmented-pricing stage, optimize the **Business** and **Leisure** prices directly. Each segment has its own demand curve, both prices are bounded by $300, and the **combined Business + Leisure demand cannot exceed the hotel's 120-room capacity**.
        """
    )

st.markdown(
    "### How this assignment will be graded"
)
st.markdown(
    "The Streamlit lab does **not** assign your grade. Submit both the **verified Streamlit PDF** and your "
    "**Excel Solver workbook**. The Excel file provides evidence of your model formulation and analysis; "
    "the verified PDF documents your submitted decisions and benchmark comparisons."
)
rubric_df = pd.DataFrame(
    [
        ["Excel model formulation", "Correct demand, revenue, decision variables, objective, and constraints", "3"],
        ["Baseline variable-pricing solution", "Solver model is used appropriately and results are entered/tested in the lab", "2"],
        ["Capacity and demand shocks", "Model is updated appropriately and revised pricing decisions are supported by analysis", "2"],
        ["Segmented pricing", "Business and Leisure demand functions and price optimization are modeled correctly", "2"],
        ["Completion and documentation", "Verified PDF and complete Excel workbook are submitted", "1"],
    ],
    columns=["Component", "Evidence", "Points"],
)
render_styled_table(rubric_df, first_col_left=True)
st.caption(
    "Conceptual 'Think About It' questions are included for learning and are not graded. "
    "Predictions are recorded in the PDF but are not scored automatically."
)

with st.sidebar:
    st.header("Student")
    student_name = st.text_input("Name", key="lab_student_name")
    student_id = st.text_input("Student ID", key="lab_student_id", help="Required to generate the final PDF report.")
    course_section = st.text_input("Section (optional)", key="lab_section")
    attempts_used = completed_attempts(student_id) if student_id.strip() else 0
    if student_id.strip():
        st.caption(f"Completed attempts: {attempts_used} of {MAX_ATTEMPTS} · Remaining: {max(0, MAX_ATTEMPTS-attempts_used)}")
    st.divider()
    st.markdown("### Lab sequence")
    st.markdown(
        """
        1. Predict weekly pricing pattern  
        2. Build and benchmark prices  
        3. Predict + test a capacity shock  
        4. Predict + test a demand shock  
        5. Predict + test segmented pricing  
        6. Review conceptual takeaways  
        7. Download verified results
        """
    )
    st.divider()
    if st.button("Start Over / New Attempt", type="secondary"):
        reset_lab()
        initialize_state()
        st.rerun()

# -----------------------------
# Stage 1
# -----------------------------

st.header("Stage 1 · Predict, Then Build Your Weekly Pricing Strategy")
st.write(
    "The hotel has **120 rooms available each night**. Before touching a room-rate slider, "
    "use the daily logit demand parameters to predict which days should support the highest room rates."
)

render_styled_table(
    BASELINE.rename(
        columns={
            "D": "Market Size (D)",
            "a": "Position Parameter (a)",
            "b": "Price Sensitivity (b)",
        }
    ),
    formats={
        "Market Size (D)": "{:,.0f}",
        "Position Parameter (a)": "{:.1f}",
        "Price Sensitivity (b)": "{:.3f}",
    },
)

if not st.session_state["lab_stage1_prediction_locked"]:
    stage1_choice = st.radio(
        "Which three days do you expect to support the highest optimized room rates?",
        [
            "Monday, Tuesday, Wednesday",
            "Thursday, Friday, Saturday",
            "Sunday, Friday, Saturday",
            "Prices should be approximately equal across all days",
        ],
        index=None,
        key="lab_stage1_choice_widget",
    )
    if st.button("Submit Weekly Pricing Prediction", type="primary", disabled=stage1_choice is None):
        st.session_state["lab_stage1_prediction"] = stage1_choice
        st.session_state["lab_stage1_prediction_locked"] = True
        st.rerun()
else:
    st.success(f"Prediction recorded: {st.session_state['lab_stage1_prediction']}")
    st.caption("Prediction recorded. This is a learning checkpoint and is not graded; you will test it against the optimizer later.")

if st.session_state["lab_stage1_prediction_locked"]:
    st.info(
        "Now solve the weekly pricing problem independently. Excel Solver is recommended. "
        "Use the sliders to enter and test the prices supported by your analysis, refine them if needed, "
        "and lock your final strategy when you are satisfied."
    )

    price_cols = st.columns(4)
    manual_prices = {}
    for i, day in enumerate(DAYS):
        with price_cols[i % 4]:
            manual_prices[day] = st.slider(
                day,
                min_value=0.0,
                max_value=300.0,
                value=150.0,
                step=0.01,
                key=f"price_{day}",
                disabled=st.session_state["lab_stage1_locked"],
            )

    manual_df = evaluate_schedule(manual_prices)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Weekly Revenue", money(manual_df["Revenue"].sum()))
    m2.metric("Weekly Occupied Rooms", f"{manual_df['Rooms Sold'].sum():,.0f}")
    m3.metric("Average Room Rate", money2(manual_df["Price"].mean()))
    m4.metric("Nights with Excess Demand", int((manual_df["Unserved Demand"] > 0).sum()))

    display_manual = manual_df[
        ["Day", "Price", "Expected Demand", "Rooms Sold", "Capacity Utilization", "Revenue"]
    ].copy()
    display_manual["Capacity Utilization"] = display_manual["Capacity Utilization"] * 100
    render_styled_table(
        display_manual,
        formats={
            "Price": "${:,.2f}",
            "Expected Demand": "{:,.0f}",
            "Rooms Sold": "{:,.0f}",
            "Capacity Utilization": "{:,.1f}%",
            "Revenue": "${:,.0f}",
        },
    )

    if not st.session_state["lab_stage1_locked"]:
        if st.button("Lock My Weekly Room-Rate Strategy", type="primary"):
            st.session_state["lab_stage1_locked"] = True
            st.session_state["lab_manual_schedule"] = manual_df.copy()
            st.rerun()
    else:
        st.success("Your original weekly room-rate strategy is locked.")
else:
    st.warning("Submit the prediction above to unlock the weekly pricing controls.")

# -----------------------------
# Stage 2
# -----------------------------

st.header("Stage 2 · Optimization Benchmark")
if not st.session_state["lab_stage1_locked"]:
    st.warning("Lock your Stage 1 pricing strategy first.")
elif not reveal_is_unlocked():
    st.info("Your Stage 1 strategy is recorded. The optimized benchmark is locked until **November 5, 2026**. Continue to Stage 3 using your own Solver analysis.")
    st.markdown("**Think About It:** If a revenue-maximizing solution leaves some rooms unsold, why might lowering the price to fill every room reduce total revenue?")
else:
    if not st.session_state["lab_stage2_revealed"]:
        if st.button("Reveal Weekly Optimization Benchmark", type="primary"):
            st.session_state["lab_stage2_revealed"] = True
            st.rerun()
    else:
        opt_df = variable_optimum(); manual_locked = st.session_state["lab_manual_schedule"]
        compare = pd.DataFrame({"Day":DAYS,"Your Price":manual_locked["Price"].values,"Optimized Price":opt_df["Price"].values,"Your Revenue":manual_locked["Revenue"].values,"Optimized Revenue":opt_df["Revenue"].values})
        render_styled_table(compare, formats={"Your Price":"${:,.2f}","Optimized Price":"${:,.2f}","Your Revenue":"${:,.0f}","Optimized Revenue":"${:,.0f}"})
        with st.expander("Think About It · Why are some rooms intentionally left unsold?"):
            st.write("Filling every room can require a price reduction large enough to reduce total revenue. Revenue maximization and occupancy maximization are different objectives.")

# -----------------------------
# Stage 3
# -----------------------------

st.header("Stage 3 · Capacity Shock: Predict Before You Reprice")

if not st.session_state["lab_stage1_locked"]:
    st.warning("Complete and lock Stage 1 first.")
else:
    st.markdown(
        """
        **Breaking update:** A maintenance issue takes 30 rooms out of service on Saturday.  
        Saturday capacity falls from **120 to 90 rooms**.
        """
    )

    sat = BASELINE[BASELINE["Day"] == "Saturday"].iloc[0]
    baseline_opt = optimize_day(sat["D"], sat["a"], sat["b"], BASE_CAPACITY, BASE_COST)
    st.caption("Baseline Saturday capacity is 120 rooms. Use your own Solver model to establish the baseline price.")

    if not st.session_state["lab_stage3_prediction_locked"]:
        direction = st.radio(
            "Before testing prices: what do you expect to happen to the optimal Saturday price?",
            ["Increase", "Decrease", "Remain approximately unchanged"],
            index=None,
            key="lab_stage3_direction_widget",
        )
        predicted_price = st.number_input(
            "Enter your predicted new optimal Saturday room rate",
            min_value=0.0,
            max_value=300.0,
            value=None,
            step=0.01,
            key="lab_stage3_pred_price_widget",
        )
        if st.button(
            "Submit Capacity-Shock Prediction",
            type="primary",
            disabled=(direction is None or predicted_price is None),
        ):
            st.session_state["lab_stage3_direction"] = direction
            st.session_state["lab_stage3_predicted_price"] = float(predicted_price)
            st.session_state["lab_stage3_prediction_locked"] = True
            st.rerun()
    else:
        st.success(
            f"Prediction recorded: {st.session_state['lab_stage3_direction']}; "
            f"predicted price {money2(st.session_state['lab_stage3_predicted_price'])}."
        )

        c1, c2 = st.columns([1, 1.3])
        with c1:
            shock_price = st.slider(
                "Test and set your revised Saturday room rate",
                0.0,
                300.0,
                float(st.session_state["lab_stage3_predicted_price"]),
                0.01,
                key="shock_sat_price",
                disabled=st.session_state["lab_stage3_locked"],
            )
            shock_demand = float(logit_demand(shock_price, sat["D"], sat["a"], sat["b"]))
            shock_sold = min(shock_demand, 90.0)
            shock_rev = shock_price * shock_sold

            st.metric("Expected Demand", f"{shock_demand:,.0f}")
            st.metric("Rooms Sold", f"{shock_sold:,.0f}")
            st.metric("Saturday Revenue", money(shock_rev))

        with c2:
            cap_view = pd.DataFrame(
                {
                    "Metric": ["Capacity", "Expected Demand", "Rooms Sold"],
                    "Rooms": [90.0, shock_demand, shock_sold],
                }
            ).set_index("Metric")
            st.bar_chart(cap_view)

        if not st.session_state["lab_stage3_locked"]:
            if st.button("Lock My Saturday Room-Rate Decision", type="primary"):
                st.session_state["lab_stage3_locked"] = True
                st.session_state["lab_capacity_shock"] = {
                    "student_price": shock_price,
                    "expected_demand": shock_demand,
                    "rooms_sold": shock_sold,
                    "revenue": shock_rev,
                }
                st.rerun()
        else:
            st.success("Your Saturday repricing decision is locked.")
            if not reveal_is_unlocked():
                st.info("Decision recorded. The optimized response and explanation unlock on **November 5, 2026**. Continue to Stage 4.")
                st.markdown("**Think About It:** How should a reduction in available rooms affect the hotel's pricing decision when demand has not changed?")
            elif not st.session_state["lab_stage3_revealed"]:
                if st.button("Reveal Optimized Saturday Response"):
                    sat_opt_90 = optimize_day(sat["D"], sat["a"], sat["b"], 90.0, BASE_COST)
                    st.session_state["lab_capacity_shock"]["optimized_price"] = sat_opt_90["price"]
                    st.session_state["lab_capacity_shock"]["optimized_revenue"] = sat_opt_90["revenue"]
                    st.session_state["lab_stage3_revealed"] = True
                    st.rerun()
            else:
                sat_opt_90 = optimize_day(sat["D"], sat["a"], sat["b"], 90.0, BASE_COST)
                actual_direction = (
                    "Increase" if sat_opt_90["price"] > baseline_opt["price"] + 0.01
                    else "Decrease" if sat_opt_90["price"] < baseline_opt["price"] - 0.01
                    else "Remain approximately unchanged"
                )
                st.info(
                    f"Your prediction: **{st.session_state['lab_stage3_direction']}** · "
                    f"Model result: **{actual_direction}**  \n"
                    f"Optimized Saturday room rate with capacity 90: **{money_md(sat_opt_90['price'], 2)}** · "
                    f"Rooms sold: **{sat_opt_90['served']:,.0f}** · "
                    f"Revenue: **{money_md(sat_opt_90['revenue'])}**"
                )

                with st.expander("Think About It · Why can lower capacity increase the optimal price?"):
                    st.write(
                        "When capacity becomes scarcer, the hotel can serve fewer customers. A higher price can "
                        "allocate the limited rooms to customers with greater willingness to pay rather than stimulate "
                        "additional demand that the hotel cannot serve."
                    )

# -----------------------------
# Stage 4
# -----------------------------

st.header("Stage 4 · Demand Shock: Predict Before You Reprice")

if not st.session_state["lab_stage3_locked"]:
    st.warning("Complete and lock Stage 3 first.")
else:
    st.markdown(
        """
        **New Scenario — Return to Baseline Conditions**  
        The temporary Saturday maintenance issue from Stage 3 has been resolved. **Hotel capacity is again 120 rooms each night.**  

        A major convention is now scheduled near the hotel on Tuesday. For this scenario, Tuesday's market-size parameter increases from **D = 160 to D = 230**. The other Tuesday demand parameters remain unchanged, and all other days remain at their original baseline conditions.
        """
    )

    tue = BASELINE[BASELINE["Day"] == "Tuesday"].iloc[0]
    original_tue_opt = optimize_day(tue["D"], tue["a"], tue["b"], BASE_CAPACITY, BASE_COST)
    st.caption("Capacity is back to 120 rooms. Use your own Solver model to establish the original Tuesday price.")

    if not st.session_state["lab_stage4_prediction_locked"]:
        direction = st.radio(
            "Before testing prices: what do you expect to happen to the optimal Tuesday price?",
            ["Increase", "Decrease", "Remain approximately unchanged"],
            index=None,
            key="lab_stage4_direction_widget",
        )
        predicted_price = st.number_input(
            "Enter your predicted new optimal Tuesday room rate",
            min_value=0.0,
            max_value=300.0,
            value=None,
            step=0.01,
            key="lab_stage4_pred_price_widget",
        )
        if st.button(
            "Submit Demand-Shock Prediction",
            type="primary",
            disabled=(direction is None or predicted_price is None),
        ):
            st.session_state["lab_stage4_direction"] = direction
            st.session_state["lab_stage4_predicted_price"] = float(predicted_price)
            st.session_state["lab_stage4_prediction_locked"] = True
            st.rerun()
    else:
        st.success(
            f"Prediction recorded: {st.session_state['lab_stage4_direction']}; "
            f"predicted price {money2(st.session_state['lab_stage4_predicted_price'])}."
        )

        c1, c2 = st.columns([1, 1.3])
        with c1:
            tue_price = st.slider(
                "Test and set your revised Tuesday room rate",
                0.0,
                300.0,
                float(st.session_state["lab_stage4_predicted_price"]),
                0.01,
                key="shock_tue_price",
                disabled=st.session_state["lab_stage4_locked"],
            )
            tue_demand = float(logit_demand(tue_price, 230.0, tue["a"], tue["b"]))
            tue_sold = min(tue_demand, BASE_CAPACITY)
            tue_rev = tue_price * tue_sold

            st.metric("Expected Demand", f"{tue_demand:,.0f}")
            st.metric("Rooms Sold", f"{tue_sold:,.0f}")
            st.metric("Tuesday Revenue", money(tue_rev))

        with c2:
            demand_view = pd.DataFrame(
                {
                    "Metric": ["Capacity", "Expected Demand", "Rooms Sold"],
                    "Rooms": [BASE_CAPACITY, tue_demand, tue_sold],
                }
            ).set_index("Metric")
            st.bar_chart(demand_view)

        if not st.session_state["lab_stage4_locked"]:
            if st.button("Lock My Tuesday Room-Rate Decision", type="primary"):
                st.session_state["lab_stage4_locked"] = True
                st.session_state["lab_demand_shock"] = {
                    "student_price": tue_price,
                    "expected_demand": tue_demand,
                    "rooms_sold": tue_sold,
                    "revenue": tue_rev,
                }
                st.rerun()
        else:
            st.success("Your Tuesday repricing decision is locked.")
            if not reveal_is_unlocked():
                st.info("Decision recorded. The optimized response and explanation unlock on **November 5, 2026**. Continue to Stage 5.")
                st.markdown("**Think About It:** How should a larger potential market affect the optimal room rate when hotel capacity remains fixed at 120 rooms?")
            elif not st.session_state["lab_stage4_revealed"]:
                if st.button("Reveal Optimized Tuesday Response"):
                    tue_opt_230 = optimize_day(230.0, tue["a"], tue["b"], BASE_CAPACITY, BASE_COST)
                    st.session_state["lab_demand_shock"]["optimized_price"] = tue_opt_230["price"]
                    st.session_state["lab_demand_shock"]["optimized_revenue"] = tue_opt_230["revenue"]
                    st.session_state["lab_stage4_revealed"] = True
                    st.rerun()
            else:
                tue_opt_230 = optimize_day(230.0, tue["a"], tue["b"], BASE_CAPACITY, BASE_COST)
                actual_direction = (
                    "Increase" if tue_opt_230["price"] > original_tue_opt["price"] + 0.01
                    else "Decrease" if tue_opt_230["price"] < original_tue_opt["price"] - 0.01
                    else "Remain approximately unchanged"
                )
                st.info(
                    f"Your prediction: **{st.session_state['lab_stage4_direction']}** · "
                    f"Model result: **{actual_direction}**  \n"
                    f"Optimized Tuesday room rate after the demand increase: **{money_md(tue_opt_230['price'], 2)}** · "
                    f"Rooms sold: **{tue_opt_230['served']:,.0f}** · "
                    f"Revenue: **{money_md(tue_opt_230['revenue'])}**"
                )

                with st.expander("Think About It · Why can stronger demand increase the optimal price?"):
                    st.write(
                        "A larger market means more potential customers are willing to purchase at each price level. "
                        "With stronger demand, the hotel can often charge a higher price while still selling enough "
                        "rooms to maximize revenue."
                    )

# -----------------------------
# Stage 5
# -----------------------------

st.header("Stage 5 · Segmented Pricing: Predict Before You Price")

if not st.session_state["lab_stage4_locked"]:
    st.warning("Complete and lock Stage 4 first.")
else:
    st.markdown(
        """
        **New Scenario — Segmented Pricing**  
        The capacity and demand shocks from Stages 3 and 4 no longer apply. The hotel again has its normal **120-room capacity**.

        The hotel now considers two customer segments with different price sensitivities. Business and Leisure customers may be charged different room rates, but their **combined expected demand cannot exceed 120 rooms**.
        """
    )

    seg_A = {"D": 110.0, "a": 4.2, "b": -0.018}
    seg_B = {"D": 180.0, "a": 4.0, "b": -0.030}
    shared_capacity = 120.0

    render_styled_table(
        pd.DataFrame([
            {
                "Segment": "A - Business",
                "Market Size (D)": seg_A["D"],
                "Position Parameter (a)": seg_A["a"],
                "Price Sensitivity (b)": seg_A["b"],
                "Maximum Price": PRICE_MAX,
            },
            {
                "Segment": "B - Leisure",
                "Market Size (D)": seg_B["D"],
                "Position Parameter (a)": seg_B["a"],
                "Price Sensitivity (b)": seg_B["b"],
                "Maximum Price": PRICE_MAX,
            },
        ]),
        formats={
            "Market Size (D)": "{:,.0f}",
            "Position Parameter (a)": "{:.1f}",
            "Price Sensitivity (b)": "{:.3f}",
            "Maximum Price": "${:,.0f}",
        },
    )

    if not st.session_state["lab_stage5_prediction_locked"]:
        sensitivity = st.radio(
            "Which segment is more price sensitive based on the b parameter?",
            ["Segment A - Business", "Segment B - Leisure", "Both are equally price sensitive"],
            index=None,
            key="lab_stage5_sensitivity_widget",
        )
        relationship = st.radio(
            "Which segmented-pricing relationship do you expect the model to recommend?",
            ["Business Price > Leisure Price", "Business Price < Leisure Price", "Business Price = Leisure Price"],
            index=None,
            key="lab_stage5_relationship_widget",
        )
        if st.button(
            "Submit Segmented-Pricing Predictions",
            type="primary",
            disabled=(sensitivity is None or relationship is None),
        ):
            st.session_state["lab_stage5_sensitive_segment"] = sensitivity
            st.session_state["lab_stage5_price_relationship"] = relationship
            st.session_state["lab_stage5_prediction_locked"] = True
            st.rerun()
    else:
        st.success(
            f"Predictions recorded: more price sensitive = {st.session_state['lab_stage5_sensitive_segment']}; "
            f"expected relationship = {st.session_state['lab_stage5_price_relationship']}."
        )
        st.info(
            "Solve the pricing problem independently. Excel Solver is recommended. Use Business Price and "
            "Leisure Price as the decision variables; maximize total revenue; constrain Business Demand + "
            "Leisure Demand to be no more than 120 rooms; constrain each price to $0–$300; and use GRG Nonlinear."
        )

        s1, s2 = st.columns(2)
        with s1:
            pA = st.slider("Business segment room rate",0.0,300.0,225.0,0.01,key="seg_A",disabled=st.session_state["lab_stage5_locked"])
        with s2:
            pB = st.slider("Leisure segment room rate",0.0,300.0,155.0,0.01,key="seg_B",disabled=st.session_state["lab_stage5_locked"])
        qA=float(logit_demand(pA,**seg_A)); qB=float(logit_demand(pB,**seg_B)); total_seg_demand=qA+qB
        segmented_feasible=total_seg_demand <= shared_capacity + 1e-9
        seg_rev=pA*qA+pB*qB if segmented_feasible else np.nan
        compare_seg=pd.DataFrame({"Segment":["Business","Leisure","Total"],"Price":[pA,pB,np.nan],"Expected Demand":[qA,qB,total_seg_demand],"Revenue":[pA*qA,pB*qB,seg_rev]})
        render_styled_table(compare_seg,formats={"Price":"${:,.2f}","Expected Demand":"{:,.2f}","Revenue":"${:,.2f}"})
        st.caption(f"Shared hotel capacity: 120 rooms · Current total expected demand: {total_seg_demand:.2f}")
        if not segmented_feasible:
            st.warning("Combined Business + Leisure demand exceeds 120 rooms. Increase one or both prices before locking your decision.")

        if not st.session_state["lab_stage5_locked"]:
            if st.button(
                "Lock My Segmented Pricing Decisions",
                type="primary",
                disabled=(not segmented_feasible),
            ):
                st.session_state["lab_stage5_locked"] = True
                st.session_state["lab_segment_result"] = {
                    "segment_A_price": pA,
                    "segment_B_price": pB,
                    "segment_A_demand": qA,
                    "segment_B_demand": qB,
                    "total_segment_demand": total_seg_demand,
                    "segmented_revenue": float(seg_rev),
                }
                st.rerun()
        else:
            st.success("Your segmented-pricing decisions are locked.")

            if not reveal_is_unlocked():
                st.info("Decision recorded. The optimized benchmark and Think About It explanation unlock on **November 5, 2026**. You may now generate your verified report.")
                st.markdown("**Think About It:** The two customer segments have different price sensitivities but share the same 120-room capacity. How should those differences influence the two prices?")
            elif not st.session_state["lab_stage5_revealed"]:
                if st.button("Reveal Optimized Segmented Pricing"):
                    grid = np.arange(PRICE_MIN, PRICE_MAX + PRICE_STEP / 2, PRICE_STEP)

                    # Segmented benchmark. Capacity binds at the optimum. For each Business price,
                    # solve the Leisure price that makes q_A + q_B = 120, then retain the best feasible pair.
                    best_seg = None
                    for pa in grid:
                        qa = float(logit_demand(pa, **seg_A))
                        qb_target = shared_capacity - qa
                        if qb_target <= 0 or qb_target >= seg_B["D"]:
                            continue
                        log_odds = np.log(qb_target / (seg_B["D"] - qb_target))
                        pb = float((log_odds - seg_B["a"]) / seg_B["b"])
                        if pb < PRICE_MIN - 1e-9 or pb > PRICE_MAX + 1e-9:
                            continue
                        qb = float(logit_demand(pb, **seg_B))
                        total = qa + qb
                        if total <= shared_capacity + 1e-6:
                            rev = pa * qa + pb * qb
                            if best_seg is None or rev > best_seg["revenue"]:
                                best_seg = {
                                    "pA": float(pa), "pB": pb, "qA": qa, "qB": qb,
                                    "total": total, "revenue": float(rev)
                                }

                    st.session_state["lab_segment_result"]["optimized_A_price"] = best_seg["pA"]
                    st.session_state["lab_segment_result"]["optimized_B_price"] = best_seg["pB"]
                    st.session_state["lab_segment_result"]["optimized_A_demand"] = best_seg["qA"]
                    st.session_state["lab_segment_result"]["optimized_B_demand"] = best_seg["qB"]
                    st.session_state["lab_segment_result"]["optimized_total_demand"] = best_seg["total"]
                    st.session_state["lab_segment_result"]["optimized_revenue"] = best_seg["revenue"]
                    st.session_state["lab_stage5_revealed"] = True
                    st.rerun()
            else:
                seg = st.session_state["lab_segment_result"]
                actual_relationship = (
                    "Business Price > Leisure Price" if seg["optimized_A_price"] > seg["optimized_B_price"] + 0.01
                    else "Business Price < Leisure Price" if seg["optimized_A_price"] < seg["optimized_B_price"] - 0.01
                    else "Business Price = Leisure Price"
                )
                st.info(
                    f"Your price-sensitivity prediction: **{st.session_state['lab_stage5_sensitive_segment']}** · "
                    f"Model interpretation: **Segment B - Leisure**  \n"
                    f"Your expected price relationship: **{st.session_state['lab_stage5_price_relationship']}** · "
                    f"Optimized relationship: **{actual_relationship}**  \n"
                    f"Optimized segmented prices: **Business {money_md(seg['optimized_A_price'], 2)}**, "
                    f"**Leisure {money_md(seg['optimized_B_price'], 2)}**  \n"
                    f"Expected demand: **Business {seg['optimized_A_demand']:.2f} + "
                    f"Leisure {seg['optimized_B_demand']:.2f} = {seg['optimized_total_demand']:.2f} rooms** · "
                    f"Revenue: **{money_md(seg['optimized_revenue'], 2)}**"
                )
                with st.expander("Think About It · Why is the Leisure price lower?"):
                    st.write(
                        "The Leisure segment has the larger absolute price-sensitivity parameter (|b|). Its demand "
                        "falls more rapidly as price increases, so its revenue-maximizing price is lower. At the "
                        "same time, the two segment prices must jointly respect the hotel's 120-room capacity."
                    )

# -----------------------------
# Stage 6: Report
# -----------------------------

st.header("Stage 6 · Generate Your Verified Pricing Lab Report")

if not st.session_state["lab_stage5_locked"]:
    st.warning("Complete and lock all prior stages before generating your report.")
else:
    opt_df = variable_optimum()
    manual_locked = st.session_state["lab_manual_schedule"]
    cap = st.session_state["lab_capacity_shock"]
    dem = st.session_state["lab_demand_shock"]
    seg = st.session_state["lab_segment_result"]
    predictions = {
        "stage1_high_days": st.session_state["lab_stage1_prediction"],
        "stage3_direction": st.session_state["lab_stage3_direction"],
        "stage3_predicted_price": st.session_state["lab_stage3_predicted_price"],
        "stage4_direction": st.session_state["lab_stage4_direction"],
        "stage4_predicted_price": st.session_state["lab_stage4_predicted_price"],
        "stage5_sensitive_segment": st.session_state["lab_stage5_sensitive_segment"],
        "stage5_price_relationship": st.session_state["lab_stage5_price_relationship"],
    }

    summary_rows = [
        ["Student", student_name or "Not entered", ""],
        ["Student ID", student_id or "Not entered", ""],
        ["Section", course_section or "", ""],
        ["Your baseline weekly revenue", manual_locked["Revenue"].sum(), "USD"],
        ["Saturday capacity-shock room rate", cap["student_price"], "USD"],
        ["Saturday capacity-shock revenue", cap["revenue"], "USD"],
        ["Tuesday demand-shock room rate", dem["student_price"], "USD"],
        ["Tuesday demand-shock revenue", dem["revenue"], "USD"],
        ["Your segmented-pricing revenue", seg["segmented_revenue"], "USD"],
    ]
    summary_df = pd.DataFrame(summary_rows, columns=["Metric", "Value", "Unit"])
    render_styled_table(summary_df, first_col_left=True)

    st.info(
        "The lab does not calculate your assignment grade. Submit the verified PDF and your Excel Solver "
        "workbook. Your grade will be based on the rubric shown at the beginning of the lab."
    )

    identity_ready = bool(student_name.strip()) and bool(student_id.strip())
    attempts_used = completed_attempts(student_id) if student_id.strip() else 0
    attempts_left = max(0, MAX_ATTEMPTS-attempts_used)
    if not identity_ready:
        st.warning("Enter both your name and Student ID in the sidebar before generating the final report.")
    elif attempts_left <= 0 and st.session_state["lab_pdf_bytes"] is None:
        st.error("You have used all 3 permitted completed attempts for this lab.")
    else:
        st.caption(f"Completed attempts: {attempts_used} of {MAX_ATTEMPTS}. Remaining before this submission: {attempts_left}.")

    if st.session_state["lab_pdf_bytes"] is None:
        if st.button("Generate Verified PDF Report", type="primary", disabled=(not identity_ready or attempts_left<=0)):
            generated_at = datetime.now(ZoneInfo("UTC"))
            verification_code = make_verification_code(student_id, generated_at)
            attempt_number = record_completed_attempt(student_id)
            pdf_bytes = build_pdf_report(
                student_name=student_name.strip(),
                student_id=student_id.strip(),
                course_section=course_section.strip(),
                generated_at=generated_at,
                verification_code=verification_code,
                manual_locked=manual_locked,
                opt_df=opt_df,
                cap=cap,
                dem=dem,
                seg=seg,
                predictions=predictions,
                benchmarks_visible=reveal_is_unlocked(),
                attempt_number=attempt_number,
            )
            st.session_state["lab_pdf_bytes"] = pdf_bytes
            st.session_state["lab_pdf_timestamp"] = generated_at.isoformat(timespec="microseconds")
            st.session_state["lab_verification_code"] = verification_code
            st.rerun()
    else:
        generated_at = datetime.fromisoformat(st.session_state["lab_pdf_timestamp"])
        generated_eastern = generated_at.astimezone(EASTERN_TZ)
        st.success(
            f"Verified report generated at {generated_eastern.strftime('%Y-%m-%d %H:%M:%S.%f %Z')}. "
            f"Verification code: {st.session_state['lab_verification_code']}"
        )

        filename_name = student_name.strip().replace(" ", "_") if student_name.strip() else "student"
        st.download_button(
            "Download Verified Pricing Lab PDF",
            data=st.session_state["lab_pdf_bytes"],
            file_name=f"{filename_name}_pricing_lab_verified.pdf",
            mime="application/pdf",
            type="primary",
        )

        st.caption(
            "The timestamp is fixed when the report is generated; downloading the same report again does not change it. "
            "Generating another verified report after starting over counts as another attempt. Maximum: 3 completed attempts."
        )

st.divider()
st.caption(
    "The lab uses the module's logit price-response model in a hotel revenue-management setting. "
    "Students should solve each pricing problem independently, use the sliders to enter and test decisions, "
    "and lock each decision. Benchmarks remain hidden until November 5, 2026."
)
