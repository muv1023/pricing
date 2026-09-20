import io
import hashlib
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

BASELINE = pd.DataFrame(
    {
        "Day": DAYS,
        "D": [4500, 1500, 1400, 1500, 2000, 4100, 5300],
        "a": [5, 4, 3, 2, 3, 4, 5],
        "b": [-0.5, -0.4, -0.3, -0.2, -0.3, -0.4, -0.5],
    }
)

BASE_CAPACITY = 1000.0
BASE_COST = 0.0
PRICE_MIN = 0.0
PRICE_MAX = 50.0
PRICE_STEP = 0.25


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
                "Tickets Sold": served,
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
                "Tickets Sold": o["served"],
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


def make_verification_code(student_id, generated_at):
    """Create a short deterministic fingerprint from student ID + exact timestamp."""
    payload = f"{student_id.strip()}|{generated_at.isoformat(timespec='microseconds')}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12].upper()


def build_pdf_report(student_name, student_id, course_section, generated_at, verification_code,
                     manual_locked, opt_df, cap, dem, seg, predictions):
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
        ["Section", course_section or "-", "Generated", time_display],
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

    story.append(Paragraph("Stage 1 - Weekly Pricing Prediction and Strategy", styles["Section"]))
    story.append(Paragraph(
        f"Pre-analysis prediction - three days expected to support the highest prices: <b>{predictions['stage1_high_days']}</b>",
        styles["BodySmall"],
    ))
    manual_rows = [["Day", "Price", "Expected Demand", "Tickets Sold", "Capacity Used", "Revenue"]]
    for _, r in manual_locked.iterrows():
        manual_rows.append([
            r["Day"], fmt_money(r["Price"]), fmt_num(r["Expected Demand"]),
            fmt_num(r["Tickets Sold"]), f"{100*r['Capacity Utilization']:.1f}%", fmt_money(r["Revenue"])
        ])
    story.append(styled_table(manual_rows, [0.9*inch, 0.8*inch, 1.15*inch, 1.0*inch, 1.05*inch, 1.0*inch]))
    story.append(Spacer(1, 5))
    story.append(Paragraph(
        f"Student weekly revenue: <b>{fmt_money(manual_locked['Revenue'].sum())}</b> | "
        f"Weekly attendance: <b>{fmt_num(manual_locked['Tickets Sold'].sum())}</b>",
        styles["BodySmall"],
    ))

    story.append(Paragraph("Stage 2 - Optimization Benchmark", styles["Section"]))
    benchmark_rows = [["Day", "Student Price", "Optimized Price", "Student Revenue", "Optimized Revenue"]]
    for i, day in enumerate(DAYS):
        benchmark_rows.append([
            day,
            fmt_money(manual_locked.iloc[i]["Price"]),
            fmt_money(opt_df.iloc[i]["Price"]),
            fmt_money(manual_locked.iloc[i]["Revenue"]),
            fmt_money(opt_df.iloc[i]["Revenue"]),
        ])
    story.append(styled_table(benchmark_rows, [1.0*inch, 1.05*inch, 1.05*inch, 1.2*inch, 1.2*inch]))
    story.append(Spacer(1, 5))
    story.append(Paragraph(
        f"Optimized weekly revenue: <b>{fmt_money(opt_df['Revenue'].sum())}</b> | "
        f"Difference from student strategy: <b>{fmt_money(opt_df['Revenue'].sum() - manual_locked['Revenue'].sum())}</b>",
        styles["BodySmall"],
    ))

    story.append(PageBreak())
    story.append(Paragraph("Stage 3 - Capacity Shock", styles["Section"]))
    cap_rows = [
        ["Pre-analysis prediction", "Student Response"],
        ["Expected direction of optimal price", predictions["stage3_direction"]],
        ["Predicted optimal Saturday price", fmt_money(predictions["stage3_predicted_price"])],
    ]
    story.append(styled_table(cap_rows, [2.8*inch, 2.6*inch]))
    story.append(Spacer(1, 5))
    cap_result_rows = [
        ["Scenario", "Student Price", "Expected Demand", "Tickets Sold", "Revenue", "Optimized Price"],
        ["Saturday capacity = 700", fmt_money(cap["student_price"]), fmt_num(cap["expected_demand"]),
         fmt_num(cap["tickets_sold"]), fmt_money(cap["revenue"]), fmt_money(cap["optimized_price"])],
    ]
    story.append(styled_table(cap_result_rows, [1.55*inch, 0.95*inch, 1.0*inch, 0.9*inch, 0.95*inch, 1.0*inch], font_size=7.5))

    story.append(Paragraph("Stage 4 - Demand Shock", styles["Section"]))
    dem_rows = [
        ["Pre-analysis prediction", "Student Response"],
        ["Expected direction of optimal price", predictions["stage4_direction"]],
        ["Predicted optimal Tuesday price", fmt_money(predictions["stage4_predicted_price"])],
    ]
    story.append(styled_table(dem_rows, [2.8*inch, 2.6*inch]))
    story.append(Spacer(1, 5))
    dem_result_rows = [
        ["Scenario", "Student Price", "Expected Demand", "Tickets Sold", "Revenue", "Optimized Price"],
        ["Tuesday D = 1,900", fmt_money(dem["student_price"]), fmt_num(dem["expected_demand"]),
         fmt_num(dem["tickets_sold"]), fmt_money(dem["revenue"]), fmt_money(dem["optimized_price"])],
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
    seg_rows = [
        ["Strategy", "Segment A Price", "Segment B Price", "Revenue"],
        ["One common price", fmt_money(seg["common_price"]), fmt_money(seg["common_price"]), fmt_money(seg["common_revenue"])],
        ["Student segmented prices", fmt_money(seg["segment_A_price"]), fmt_money(seg["segment_B_price"]), fmt_money(seg["segmented_revenue"])],
        ["Optimized segmented prices", fmt_money(seg["optimized_A_price"]), fmt_money(seg["optimized_B_price"]), fmt_money(seg["optimized_revenue"])],
    ]
    story.append(styled_table(seg_rows, [2.0*inch, 1.35*inch, 1.35*inch, 1.35*inch]))

    story.append(Spacer(1, 18))
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
        Price the business yourself first. Then benchmark your decisions against optimization.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Student")
    student_name = st.text_input("Name", key="lab_student_name")
    student_id = st.text_input("Student ID", key="lab_student_id", help="Required to generate the final PDF report.")
    course_section = st.text_input("Section (optional)", key="lab_section")
    st.divider()
    st.markdown("### Lab sequence")
    st.markdown(
        """
        1. Predict weekly pricing pattern  
        2. Build and benchmark prices  
        3. Predict + test a capacity shock  
        4. Predict + test a demand shock  
        5. Predict + test segmented pricing  
        6. Download verified results
        """
    )
    st.divider()
    if st.button("Reset entire lab", type="secondary"):
        reset_lab()
        initialize_state()
        st.rerun()

# -----------------------------
# Stage 1
# -----------------------------

st.header("Stage 1 · Predict, Then Build Your Weekly Pricing Strategy")
st.write(
    "The theme park can serve up to **1,000 customers per day**. Before touching a price slider, "
    "use the daily logit demand parameters to predict which days should support the highest prices."
)

st.dataframe(
    BASELINE.rename(columns={"D": "Market Size (D)", "a": "a", "b": "b"}),
    use_container_width=True,
    hide_index=True,
)

if not st.session_state["lab_stage1_prediction_locked"]:
    stage1_choice = st.radio(
        "Which three days do you expect to support the highest optimized ticket prices?",
        [
            "Monday, Tuesday, Wednesday",
            "Sunday, Friday, Saturday",
            "Tuesday, Thursday, Saturday",
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
    st.caption("The lab will not grade this prediction yet. You will test it through your pricing decisions and the optimizer.")

if st.session_state["lab_stage1_prediction_locked"]:
    st.info(
        "Now build your weekly pricing strategy. Use the demand parameters and the live dashboard, "
        "but do not run an optimizer outside the lab before making your decisions."
    )

    price_cols = st.columns(4)
    manual_prices = {}
    for i, day in enumerate(DAYS):
        with price_cols[i % 4]:
            manual_prices[day] = st.slider(
                day,
                min_value=0.0,
                max_value=50.0,
                value=15.0,
                step=0.5,
                key=f"price_{day}",
                disabled=st.session_state["lab_stage1_locked"],
            )

    manual_df = evaluate_schedule(manual_prices)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Weekly Revenue", money(manual_df["Revenue"].sum()))
    m2.metric("Weekly Attendance", f"{manual_df['Tickets Sold'].sum():,.0f}")
    m3.metric("Average Ticket Price", money2(manual_df["Price"].mean()))
    m4.metric("Days with Excess Demand", int((manual_df["Unserved Demand"] > 0).sum()))

    display_manual = manual_df[
        ["Day", "Price", "Expected Demand", "Tickets Sold", "Capacity Utilization", "Revenue"]
    ].copy()
    display_manual["Capacity Utilization"] = display_manual["Capacity Utilization"] * 100
    st.dataframe(
        display_manual.style.format(
            {
                "Price": "${:,.2f}",
                "Expected Demand": "{:,.0f}",
                "Tickets Sold": "{:,.0f}",
                "Capacity Utilization": "{:,.1f}%",
                "Revenue": "${:,.0f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    if not st.session_state["lab_stage1_locked"]:
        if st.button("Lock My Weekly Pricing Strategy", type="primary"):
            st.session_state["lab_stage1_locked"] = True
            st.session_state["lab_manual_schedule"] = manual_df.copy()
            st.rerun()
    else:
        st.success("Your original weekly pricing strategy is locked.")
else:
    st.warning("Submit the prediction above to unlock the weekly pricing controls.")

# -----------------------------
# Stage 2
# -----------------------------

st.header("Stage 2 · Benchmark Against Optimization")

if not st.session_state["lab_stage1_locked"]:
    st.warning("Lock your Stage 1 pricing strategy before benchmarking it.")
else:
    if not st.session_state["lab_stage2_revealed"]:
        st.write(
            "You have made the pricing decision. Now you may reveal the optimized variable-pricing benchmark."
        )
        if st.button("Run Pricing Optimizer", type="primary"):
            st.session_state["lab_stage2_revealed"] = True
            st.rerun()
    else:
        opt_df = variable_optimum()
        manual_locked = st.session_state["lab_manual_schedule"]

        manual_revenue = float(manual_locked["Revenue"].sum())
        opt_revenue = float(opt_df["Revenue"].sum())
        improvement = opt_revenue - manual_revenue

        b1, b2, b3 = st.columns(3)
        b1.metric("Your Weekly Revenue", money(manual_revenue))
        b2.metric("Optimized Weekly Revenue", money(opt_revenue))
        b3.metric("Revenue Difference", money(improvement))

        compare = pd.DataFrame(
            {
                "Day": DAYS,
                "Your Price": manual_locked["Price"].values,
                "Optimized Price": opt_df["Price"].values,
                "Your Revenue": manual_locked["Revenue"].values,
                "Optimized Revenue": opt_df["Revenue"].values,
            }
        )
        st.dataframe(
            compare.style.format(
                {
                    "Your Price": "${:,.2f}",
                    "Optimized Price": "${:,.2f}",
                    "Your Revenue": "${:,.0f}",
                    "Optimized Revenue": "${:,.0f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.bar_chart(compare.set_index("Day")[["Your Price", "Optimized Price"]])

        top3 = ", ".join(opt_df.sort_values("Price", ascending=False).head(3)["Day"].tolist())
        expected_stage1 = "Sunday, Friday, Saturday"
        supported = st.session_state["lab_stage1_prediction"] == expected_stage1
        st.info(
            f"Your pre-analysis prediction: **{st.session_state['lab_stage1_prediction']}**  \n"
            f"Three highest optimized-price days in this model: **{top3}**  \n"
            f"Prediction supported by the model: **{'Yes' if supported else 'No'}**"
        )

# -----------------------------
# Stage 3
# -----------------------------

st.header("Stage 3 · Capacity Shock: Predict Before You Reprice")

if not st.session_state["lab_stage2_revealed"]:
    st.warning("Complete Stage 2 first.")
else:
    st.markdown(
        """
        **Breaking update:** One major attraction will be closed on Saturday for maintenance.  
        Saturday capacity falls from **1,000 to 700 customers**.
        """
    )

    sat = BASELINE[BASELINE["Day"] == "Saturday"].iloc[0]
    baseline_opt = optimize_day(sat["D"], sat["a"], sat["b"], BASE_CAPACITY, BASE_COST)
    st.metric("Original Optimized Saturday Price", money2(baseline_opt["price"]))

    if not st.session_state["lab_stage3_prediction_locked"]:
        direction = st.radio(
            "Before testing prices: what do you expect to happen to the optimal Saturday price?",
            ["Increase", "Decrease", "Remain approximately unchanged"],
            index=None,
            key="lab_stage3_direction_widget",
        )
        predicted_price = st.number_input(
            "Enter your predicted new optimal Saturday price",
            min_value=0.0,
            max_value=50.0,
            value=None,
            step=0.50,
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
                "Test and set your revised Saturday price",
                0.0,
                50.0,
                float(st.session_state["lab_stage3_predicted_price"]),
                0.5,
                key="shock_sat_price",
                disabled=st.session_state["lab_stage3_locked"],
            )
            shock_demand = float(logit_demand(shock_price, sat["D"], sat["a"], sat["b"]))
            shock_sold = min(shock_demand, 700.0)
            shock_rev = shock_price * shock_sold

            st.metric("Expected Demand", f"{shock_demand:,.0f}")
            st.metric("Tickets Sold", f"{shock_sold:,.0f}")
            st.metric("Saturday Revenue", money(shock_rev))

        with c2:
            cap_view = pd.DataFrame(
                {
                    "Metric": ["Capacity", "Expected Demand", "Tickets Sold"],
                    "Customers": [700.0, shock_demand, shock_sold],
                }
            ).set_index("Metric")
            st.bar_chart(cap_view)

        if not st.session_state["lab_stage3_locked"]:
            if st.button("Lock My Saturday Repricing", type="primary"):
                st.session_state["lab_stage3_locked"] = True
                st.session_state["lab_capacity_shock"] = {
                    "student_price": shock_price,
                    "expected_demand": shock_demand,
                    "tickets_sold": shock_sold,
                    "revenue": shock_rev,
                }
                st.rerun()
        else:
            st.success("Your Saturday repricing decision is locked.")
            if not st.session_state["lab_stage3_revealed"]:
                if st.button("Reveal Optimized Saturday Response"):
                    sat_opt_700 = optimize_day(sat["D"], sat["a"], sat["b"], 700.0, BASE_COST)
                    st.session_state["lab_capacity_shock"]["optimized_price"] = sat_opt_700["price"]
                    st.session_state["lab_capacity_shock"]["optimized_revenue"] = sat_opt_700["revenue"]
                    st.session_state["lab_stage3_revealed"] = True
                    st.rerun()
            else:
                sat_opt_700 = optimize_day(sat["D"], sat["a"], sat["b"], 700.0, BASE_COST)
                actual_direction = (
                    "Increase" if sat_opt_700["price"] > baseline_opt["price"] + 0.01
                    else "Decrease" if sat_opt_700["price"] < baseline_opt["price"] - 0.01
                    else "Remain approximately unchanged"
                )
                st.info(
                    f"Your prediction: **{st.session_state['lab_stage3_direction']}** · "
                    f"Model result: **{actual_direction}**  \n"
                    f"Optimized Saturday price with capacity 700: **{money_md(sat_opt_700['price'], 2)}** · "
                    f"Tickets sold: **{sat_opt_700['served']:,.0f}** · "
                    f"Revenue: **{money_md(sat_opt_700['revenue'])}**"
                )

# -----------------------------
# Stage 4
# -----------------------------

st.header("Stage 4 · Demand Shock: Predict Before You Reprice")

if not st.session_state["lab_stage3_revealed"]:
    st.warning("Complete Stage 3 first.")
else:
    st.markdown(
        """
        **New market information:** A major convention is scheduled near the park on Tuesday.  
        For this lab, Tuesday's market-size parameter increases from **D = 1,400 to D = 1,900**.  
        The other Tuesday demand parameters remain unchanged.
        """
    )

    tue = BASELINE[BASELINE["Day"] == "Tuesday"].iloc[0]
    original_tue_opt = optimize_day(tue["D"], tue["a"], tue["b"], BASE_CAPACITY, BASE_COST)
    st.metric("Original Optimized Tuesday Price", money2(original_tue_opt["price"]))

    if not st.session_state["lab_stage4_prediction_locked"]:
        direction = st.radio(
            "Before testing prices: what do you expect to happen to the optimal Tuesday price?",
            ["Increase", "Decrease", "Remain approximately unchanged"],
            index=None,
            key="lab_stage4_direction_widget",
        )
        predicted_price = st.number_input(
            "Enter your predicted new optimal Tuesday price",
            min_value=0.0,
            max_value=50.0,
            value=None,
            step=0.50,
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
                "Test and set your revised Tuesday price",
                0.0,
                50.0,
                float(st.session_state["lab_stage4_predicted_price"]),
                0.5,
                key="shock_tue_price",
                disabled=st.session_state["lab_stage4_locked"],
            )
            tue_demand = float(logit_demand(tue_price, 1900.0, tue["a"], tue["b"]))
            tue_sold = min(tue_demand, BASE_CAPACITY)
            tue_rev = tue_price * tue_sold

            st.metric("Expected Demand", f"{tue_demand:,.0f}")
            st.metric("Tickets Sold", f"{tue_sold:,.0f}")
            st.metric("Tuesday Revenue", money(tue_rev))

        with c2:
            demand_view = pd.DataFrame(
                {
                    "Metric": ["Capacity", "Expected Demand", "Tickets Sold"],
                    "Customers": [BASE_CAPACITY, tue_demand, tue_sold],
                }
            ).set_index("Metric")
            st.bar_chart(demand_view)

        if not st.session_state["lab_stage4_locked"]:
            if st.button("Lock My Tuesday Repricing", type="primary"):
                st.session_state["lab_stage4_locked"] = True
                st.session_state["lab_demand_shock"] = {
                    "student_price": tue_price,
                    "expected_demand": tue_demand,
                    "tickets_sold": tue_sold,
                    "revenue": tue_rev,
                }
                st.rerun()
        else:
            st.success("Your Tuesday repricing decision is locked.")
            if not st.session_state["lab_stage4_revealed"]:
                if st.button("Reveal Optimized Tuesday Response"):
                    tue_opt_1900 = optimize_day(1900.0, tue["a"], tue["b"], BASE_CAPACITY, BASE_COST)
                    st.session_state["lab_demand_shock"]["optimized_price"] = tue_opt_1900["price"]
                    st.session_state["lab_demand_shock"]["optimized_revenue"] = tue_opt_1900["revenue"]
                    st.session_state["lab_stage4_revealed"] = True
                    st.rerun()
            else:
                tue_opt_1900 = optimize_day(1900.0, tue["a"], tue["b"], BASE_CAPACITY, BASE_COST)
                actual_direction = (
                    "Increase" if tue_opt_1900["price"] > original_tue_opt["price"] + 0.01
                    else "Decrease" if tue_opt_1900["price"] < original_tue_opt["price"] - 0.01
                    else "Remain approximately unchanged"
                )
                st.info(
                    f"Your prediction: **{st.session_state['lab_stage4_direction']}** · "
                    f"Model result: **{actual_direction}**  \n"
                    f"Optimized Tuesday price after the demand increase: **{money_md(tue_opt_1900['price'], 2)}** · "
                    f"Tickets sold: **{tue_opt_1900['served']:,.0f}** · "
                    f"Revenue: **{money_md(tue_opt_1900['revenue'])}**"
                )

# -----------------------------
# Stage 5
# -----------------------------

st.header("Stage 5 · Segmented Pricing: Predict Before You Price")

if not st.session_state["lab_stage4_revealed"]:
    st.warning("Complete Stage 4 first.")
else:
    st.write(
        "A separate venue has two customer segments with different price sensitivities. "
        "Both use the logit demand model and share a capacity of 3,000."
    )

    seg_A = {"D": 9000.0, "a": 1.0, "b": -0.2}
    seg_B = {"D": 5000.0, "a": 1.0, "b": -0.5}
    shared_capacity = 3000.0

    st.dataframe(
        pd.DataFrame([
            {"Segment": "A", "D": seg_A["D"], "a": seg_A["a"], "b": seg_A["b"]},
            {"Segment": "B", "D": seg_B["D"], "a": seg_B["a"], "b": seg_B["b"]},
        ]),
        use_container_width=True,
        hide_index=True,
    )

    if not st.session_state["lab_stage5_prediction_locked"]:
        sensitivity = st.radio(
            "Which segment is more price sensitive based on the b parameter?",
            ["Segment A", "Segment B", "Both are equally price sensitive"],
            index=None,
            key="lab_stage5_sensitivity_widget",
        )
        relationship = st.radio(
            "Which segmented-pricing relationship do you expect the model to recommend?",
            ["Price A > Price B", "Price A < Price B", "Price A = Price B"],
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

        s1, s2, s3 = st.columns(3)
        with s1:
            common_p = st.slider(
                "One price for everyone",
                0.0,
                20.0,
                8.0,
                0.25,
                key="seg_common",
                disabled=st.session_state["lab_stage5_locked"],
            )
        with s2:
            pA = st.slider(
                "Segment A price",
                0.0,
                20.0,
                10.0,
                0.25,
                key="seg_A",
                disabled=st.session_state["lab_stage5_locked"],
            )
        with s3:
            pB = st.slider(
                "Segment B price",
                0.0,
                20.0,
                6.0,
                0.25,
                key="seg_B",
                disabled=st.session_state["lab_stage5_locked"],
            )

        qA_c = float(logit_demand(common_p, **seg_A))
        qB_c = float(logit_demand(common_p, **seg_B))
        total_c = qA_c + qB_c
        scale_c = min(1.0, shared_capacity / total_c) if total_c > 0 else 0
        soldA_c = qA_c * scale_c
        soldB_c = qB_c * scale_c
        common_rev = common_p * (soldA_c + soldB_c)

        qA = float(logit_demand(pA, **seg_A))
        qB = float(logit_demand(pB, **seg_B))
        total = qA + qB
        scale = min(1.0, shared_capacity / total) if total > 0 else 0
        soldA = qA * scale
        soldB = qB * scale
        seg_rev = pA * soldA + pB * soldB

        compare_seg = pd.DataFrame(
            {
                "Strategy": ["One Price", "Segmented Prices"],
                "Segment A Price": [common_p, pA],
                "Segment B Price": [common_p, pB],
                "Total Tickets Sold": [soldA_c + soldB_c, soldA + soldB],
                "Revenue": [common_rev, seg_rev],
            }
        )
        st.dataframe(
            compare_seg.style.format(
                {
                    "Segment A Price": "${:,.2f}",
                    "Segment B Price": "${:,.2f}",
                    "Total Tickets Sold": "{:,.0f}",
                    "Revenue": "${:,.0f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        if not st.session_state["lab_stage5_locked"]:
            if st.button("Lock My Segmented Pricing Decisions", type="primary"):
                st.session_state["lab_stage5_locked"] = True
                st.session_state["lab_segment_result"] = {
                    "common_price": common_p,
                    "common_revenue": common_rev,
                    "segment_A_price": pA,
                    "segment_B_price": pB,
                    "segmented_revenue": seg_rev,
                }
                st.rerun()
        else:
            st.success("Your segmented-pricing decisions are locked.")

            if not st.session_state["lab_stage5_revealed"]:
                if st.button("Reveal Optimized Segmented Pricing"):
                    grid = np.arange(0.0, 20.0 + 0.25, 0.25)
                    best_common = None
                    for p in grid:
                        qa = float(logit_demand(p, **seg_A))
                        qb = float(logit_demand(p, **seg_B))
                        t = qa + qb
                        sc = min(1.0, shared_capacity / t) if t > 0 else 0
                        rev = p * (qa + qb) * sc
                        if best_common is None or rev > best_common["revenue"]:
                            best_common = {"price": p, "revenue": rev}

                    best_seg = None
                    for pa in grid:
                        qa = float(logit_demand(pa, **seg_A))
                        for pb in grid:
                            qb = float(logit_demand(pb, **seg_B))
                            t = qa + qb
                            sc = min(1.0, shared_capacity / t) if t > 0 else 0
                            rev = pa * qa * sc + pb * qb * sc
                            if best_seg is None or rev > best_seg["revenue"]:
                                best_seg = {"pA": pa, "pB": pb, "revenue": rev}

                    st.session_state["lab_segment_result"]["optimized_common_price"] = best_common["price"]
                    st.session_state["lab_segment_result"]["optimized_common_revenue"] = best_common["revenue"]
                    st.session_state["lab_segment_result"]["optimized_A_price"] = best_seg["pA"]
                    st.session_state["lab_segment_result"]["optimized_B_price"] = best_seg["pB"]
                    st.session_state["lab_segment_result"]["optimized_revenue"] = best_seg["revenue"]
                    st.session_state["lab_stage5_revealed"] = True
                    st.rerun()
            else:
                seg = st.session_state["lab_segment_result"]
                actual_relationship = (
                    "Price A > Price B" if seg["optimized_A_price"] > seg["optimized_B_price"] + 0.01
                    else "Price A < Price B" if seg["optimized_A_price"] < seg["optimized_B_price"] - 0.01
                    else "Price A = Price B"
                )
                st.info(
                    f"Your price-sensitivity prediction: **{st.session_state['lab_stage5_sensitive_segment']}** · "
                    f"Model interpretation: **Segment B**  \n"
                    f"Your expected price relationship: **{st.session_state['lab_stage5_price_relationship']}** · "
                    f"Optimized relationship: **{actual_relationship}**  \n"
                    f"Optimized common price: **{money_md(seg['optimized_common_price'], 2)}** · "
                    f"Revenue: **{money_md(seg['optimized_common_revenue'])}**  \n"
                    f"Optimized segmented prices: **A {money_md(seg['optimized_A_price'], 2)}**, "
                    f"**B {money_md(seg['optimized_B_price'], 2)}** · "
                    f"Revenue: **{money_md(seg['optimized_revenue'])}**"
                )

# -----------------------------
# Stage 6: Report
# -----------------------------

st.header("Stage 6 · Generate Your Verified Pricing Lab Report")

if not st.session_state["lab_stage5_revealed"]:
    st.warning("Complete all prior stages before generating your report.")
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
        ["Optimized baseline weekly revenue", opt_df["Revenue"].sum(), "USD"],
        ["Saturday capacity-shock price", cap["student_price"], "USD"],
        ["Saturday capacity-shock revenue", cap["revenue"], "USD"],
        ["Tuesday demand-shock price", dem["student_price"], "USD"],
        ["Tuesday demand-shock revenue", dem["revenue"], "USD"],
        ["One-price segmented scenario revenue", seg["common_revenue"], "USD"],
        ["Your segmented-pricing revenue", seg["segmented_revenue"], "USD"],
    ]
    summary_df = pd.DataFrame(summary_rows, columns=["Metric", "Value", "Unit"])
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.info(
        "The final PDF records the exact server generation time in U.S. Eastern Time and UTC. "
        "It also creates a verification code from the Student ID and exact timestamp."
    )

    identity_ready = bool(student_name.strip()) and bool(student_id.strip())
    if not identity_ready:
        st.warning("Enter both your name and Student ID in the sidebar before generating the final report.")

    if st.session_state["lab_pdf_bytes"] is None:
        if st.button("Generate Verified PDF Report", type="primary", disabled=not identity_ready):
            generated_at = datetime.now(ZoneInfo("UTC"))
            verification_code = make_verification_code(student_id, generated_at)
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
            "To create a new timestamp/code, reset the lab and complete a new run."
        )

st.divider()
st.caption(
    "The lab uses the module's logit price-response model and the theme-park parameters from the pricing assignment. "
    "The purpose is to require a conceptual prediction before each pricing experiment, then compare the student's decision with an optimization benchmark."
)
