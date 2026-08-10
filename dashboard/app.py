import os
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─── Page Configuration ─────────────────────────────────────────────────────

st.set_page_config(
    page_title="LLM Eval Dashboard",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Paths ───────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
HISTORY_CSV = os.path.join(RESULTS_DIR, "history.csv")
EVAL_RESULTS_JSON = os.path.join(RESULTS_DIR, "eval_results.json")

# ─── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global Styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Hero Header */
    .hero-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        border-radius: 16px;
        padding: 32px 40px;
        margin-bottom: 24px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    .hero-header h1 {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #a78bfa, #818cf8, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    .hero-header p {
        color: #94a3b8;
        font-size: 0.95rem;
        margin: 0;
    }

    /* Status Badge */
    .status-badge {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
        margin-top: 8px;
    }
    .status-pass {
        background: linear-gradient(135deg, #065f46, #047857);
        color: #6ee7b7;
        border: 1px solid #10b981;
    }
    .status-fail {
        background: linear-gradient(135deg, #7f1d1d, #991b1b);
        color: #fca5a5;
        border: 1px solid #ef4444;
    }

    /* KPI Card */
    .kpi-card {
        background: linear-gradient(145deg, #1e1b4b, #1e293b);
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.06);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(99, 102, 241, 0.15);
    }
    .kpi-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 800;
        margin: 4px 0;
    }
    .kpi-threshold {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 4px;
    }
    .kpi-pass { color: #34d399; }
    .kpi-fail { color: #f87171; }
    .kpi-neutral { color: #818cf8; }

    /* Section Headers */
    .section-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #e2e8f0;
        margin: 32px 0 16px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid rgba(99, 102, 241, 0.3);
    }

    /* Chart Container */
    .chart-container {
        background: rgba(15, 23, 42, 0.6);
        border-radius: 14px;
        padding: 16px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Threshold Table */
    .threshold-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .threshold-table th {
        background: linear-gradient(135deg, #1e1b4b, #312e81);
        color: #c4b5fd;
        padding: 12px 16px;
        text-align: left;
        font-weight: 600;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .threshold-table td {
        padding: 10px 16px;
        background: rgba(15, 23, 42, 0.5);
        color: #e2e8f0;
        font-size: 0.9rem;
        border-top: 1px solid rgba(255, 255, 255, 0.04);
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─── Helper Functions ────────────────────────────────────────────────────────

def load_history() -> pd.DataFrame:
    """Load the history CSV file."""
    if not os.path.isfile(HISTORY_CSV):
        return pd.DataFrame()
    try:
        df = pd.read_csv(HISTORY_CSV)
        if df.empty:
            return pd.DataFrame()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    except Exception:
        return pd.DataFrame()


def load_eval_results() -> dict:
    """Load the latest evaluation results JSON."""
    if not os.path.isfile(EVAL_RESULTS_JSON):
        return {}
    try:
        with open(EVAL_RESULTS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def kpi_card(label: str, value: str, status: str = "neutral", threshold_text: str = "") -> str:
    """Generate HTML for a KPI card."""
    css_class = f"kpi-{status}"
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value {css_class}">{value}</div>
        <div class="kpi-threshold">{threshold_text}</div>
    </div>
    """


# ─── Main Dashboard ─────────────────────────────────────────────────────────

def main():
    # Load data
    history_df = load_history()
    eval_results = load_eval_results()

    # ── Hero Header ──────────────────────────────────────────────────────
    last_run = "No runs yet"
    overall_status = "N/A"
    status_class = ""

    if not history_df.empty:
        last_row = history_df.iloc[-1]
        last_run = last_row["timestamp"].strftime("%B %d, %Y at %H:%M UTC")
        overall_status = last_row["pass_fail"]
        status_class = "status-pass" if overall_status == "PASS" else "status-fail"

    st.markdown(f"""
    <div class="hero-header">
        <h1>🧪 LLM Evaluation Dashboard</h1>
        <p>RAG Chatbot — University Rules Q&A Pipeline</p>
        <p style="margin-top: 8px; font-size: 0.85rem; color: #64748b;">
            Last Run: {last_run}
        </p>
        <span class="status-badge {status_class}">{overall_status}</span>
    </div>
    """, unsafe_allow_html=True)

    if history_df.empty:
        st.info("📊 No evaluation data found. Run `python main.py` to generate results.")
        _render_thresholds_section()
        return

    # ── KPI Cards ────────────────────────────────────────────────────────
    last = history_df.iloc[-1]

    accuracy_val = last["accuracy"]
    hallucination_val = last["hallucination_rate"]
    latency_val = last["avg_latency"]
    cost_val = last["avg_cost"]
    failed_val = int(last["failed_count"])

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        status = "pass" if accuracy_val >= 0.7 else "fail"
        st.markdown(kpi_card("Accuracy", f"{accuracy_val:.1%}", status, "Threshold: ≥ 70%"), unsafe_allow_html=True)
    with col2:
        status = "pass" if hallucination_val <= 0.2 else "fail"
        st.markdown(kpi_card("Hallucination", f"{hallucination_val:.1%}", status, "Threshold: ≤ 20%"), unsafe_allow_html=True)
    with col3:
        status = "pass" if latency_val <= 10.0 else "fail"
        st.markdown(kpi_card("Avg Latency", f"{latency_val:.2f}s", status, "Threshold: ≤ 10s"), unsafe_allow_html=True)
    with col4:
        status = "pass" if cost_val <= 0.05 else "fail"
        st.markdown(kpi_card("Avg Cost", f"${cost_val:.4f}", status, "Threshold: ≤ $0.05"), unsafe_allow_html=True)
    with col5:
        status = "pass" if failed_val <= 5 else "fail"
        st.markdown(kpi_card("Failed Qs", str(failed_val), status, "Threshold: ≤ 5"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Historical Trend Charts ──────────────────────────────────────────
    st.markdown('<div class="section-header">📈 Historical Trends</div>', unsafe_allow_html=True)

    if len(history_df) >= 2:
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            # Accuracy over time
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(
                x=history_df["timestamp"],
                y=history_df["accuracy"],
                mode="lines+markers",
                name="Accuracy",
                line=dict(color="#818cf8", width=3),
                marker=dict(size=8, color="#a78bfa"),
                fill="tozeroy",
                fillcolor="rgba(129, 140, 248, 0.1)",
            ))
            fig_acc.add_hline(y=0.7, line_dash="dash", line_color="#f87171",
                            annotation_text="Threshold (70%)")
            fig_acc.update_layout(
                title="Accuracy Over Time",
                xaxis_title="",
                yaxis_title="Score",
                yaxis=dict(range=[0, 1]),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
                height=350,
                margin=dict(l=40, r=20, t=50, b=30),
            )
            st.plotly_chart(fig_acc, use_container_width=True)

            # Latency over time
            fig_lat = go.Figure()
            fig_lat.add_trace(go.Scatter(
                x=history_df["timestamp"],
                y=history_df["avg_latency"],
                mode="lines+markers",
                name="Latency",
                line=dict(color="#fb923c", width=3),
                marker=dict(size=8, color="#fdba74"),
                fill="tozeroy",
                fillcolor="rgba(251, 146, 60, 0.1)",
            ))
            fig_lat.add_hline(y=10.0, line_dash="dash", line_color="#f87171",
                            annotation_text="Threshold (10s)")
            fig_lat.update_layout(
                title="Average Latency Over Time",
                xaxis_title="",
                yaxis_title="Seconds",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
                height=350,
                margin=dict(l=40, r=20, t=50, b=30),
            )
            st.plotly_chart(fig_lat, use_container_width=True)

        with chart_col2:
            # Hallucination over time
            fig_hall = go.Figure()
            fig_hall.add_trace(go.Scatter(
                x=history_df["timestamp"],
                y=history_df["hallucination_rate"],
                mode="lines+markers",
                name="Hallucination",
                line=dict(color="#f472b6", width=3),
                marker=dict(size=8, color="#f9a8d4"),
                fill="tozeroy",
                fillcolor="rgba(244, 114, 182, 0.1)",
            ))
            fig_hall.add_hline(y=0.2, line_dash="dash", line_color="#f87171",
                              annotation_text="Threshold (20%)")
            fig_hall.update_layout(
                title="Hallucination Rate Over Time",
                xaxis_title="",
                yaxis_title="Rate",
                yaxis=dict(range=[0, 1]),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
                height=350,
                margin=dict(l=40, r=20, t=50, b=30),
            )
            st.plotly_chart(fig_hall, use_container_width=True)

            # Cost over time
            fig_cost = go.Figure()
            fig_cost.add_trace(go.Scatter(
                x=history_df["timestamp"],
                y=history_df["avg_cost"],
                mode="lines+markers",
                name="Cost",
                line=dict(color="#34d399", width=3),
                marker=dict(size=8, color="#6ee7b7"),
                fill="tozeroy",
                fillcolor="rgba(52, 211, 153, 0.1)",
            ))
            fig_cost.add_hline(y=0.05, line_dash="dash", line_color="#f87171",
                              annotation_text="Threshold ($0.05)")
            fig_cost.update_layout(
                title="Average Cost Per Query Over Time",
                xaxis_title="",
                yaxis_title="USD",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
                height=350,
                margin=dict(l=40, r=20, t=50, b=30),
            )
            st.plotly_chart(fig_cost, use_container_width=True)

    else:
        st.info("📊 At least 2 evaluation runs are needed to display trend charts.")

    # ── Per-Question Drill-Down ──────────────────────────────────────────
    st.markdown('<div class="section-header">🔍 Per-Question Results (Latest Run)</div>', unsafe_allow_html=True)

    if eval_results and "results" in eval_results:
        results_list = eval_results["results"]

        # Build a DataFrame for display
        rows = []
        for r in results_list:
            metrics = r.get("metrics", {})
            rows.append({
                "ID": r.get("id", ""),
                "Category": r.get("category", ""),
                "Question": r.get("question", "")[:80] + "..." if len(r.get("question", "")) > 80 else r.get("question", ""),
                "Accuracy": f"{metrics.get('accuracy', 0):.1%}",
                "Hallucination": f"{metrics.get('hallucination', 0):.1%}",
                "Latency": f"{metrics.get('latency', 0):.2f}s",
                "Cost": f"${metrics.get('cost', 0):.6f}",
                "Status": "❌ FAIL" if r.get("failed", False) else "✅ PASS",
            })

        results_df = pd.DataFrame(rows)
        st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True,
            height=min(len(rows) * 38 + 40, 600),
        )

        # Expandable details
        with st.expander("📋 View Full Question & Answer Details"):
            for r in results_list:
                status = "❌" if r.get("failed", False) else "✅"
                st.markdown(f"**{status} {r.get('id', '')}** — {r.get('question', '')}")
                st.markdown(f"**Ground Truth:** {r.get('ground_truth', '')}")
                st.markdown(f"**Predicted:** {r.get('predicted_answer', '')}")
                st.markdown(f"**Sources:** {', '.join(r.get('sources', []))}")
                st.divider()

    else:
        st.info("No per-question results available. Run `python main.py` first.")

    # ── Run History Table ────────────────────────────────────────────────
    st.markdown('<div class="section-header">📋 Run History</div>', unsafe_allow_html=True)

    display_df = history_df.copy()
    display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    display_df["accuracy"] = display_df["accuracy"].apply(lambda x: f"{x:.1%}")
    display_df["hallucination_rate"] = display_df["hallucination_rate"].apply(lambda x: f"{x:.1%}")
    display_df["avg_latency"] = display_df["avg_latency"].apply(lambda x: f"{x:.2f}s")
    display_df["avg_cost"] = display_df["avg_cost"].apply(lambda x: f"${x:.6f}")

    display_df.columns = ["Timestamp", "Accuracy", "Hallucination", "Latency", "Cost",
                          "Failed", "Total Qs", "Result"]

    st.dataframe(
        display_df.iloc[::-1],  # Most recent first
        use_container_width=True,
        hide_index=True,
    )

    # ── Thresholds Section ───────────────────────────────────────────────
    _render_thresholds_section()


def _render_thresholds_section():
    """Render the threshold configuration display."""
    st.markdown('<div class="section-header">⚙️ Threshold Configuration</div>', unsafe_allow_html=True)

    st.markdown("""
    <table class="threshold-table">
        <tr>
            <th>Metric</th>
            <th>Operator</th>
            <th>Threshold</th>
            <th>Description</th>
        </tr>
        <tr>
            <td>🎯 Accuracy</td>
            <td>≥</td>
            <td>70%</td>
            <td>Minimum average accuracy vs. ground truth</td>
        </tr>
        <tr>
            <td>👻 Hallucination</td>
            <td>≤</td>
            <td>20%</td>
            <td>Maximum average hallucination rate</td>
        </tr>
        <tr>
            <td>⏱️ Latency</td>
            <td>≤</td>
            <td>10.0s</td>
            <td>Maximum average latency per query</td>
        </tr>
        <tr>
            <td>💰 Cost</td>
            <td>≤</td>
            <td>$0.05</td>
            <td>Maximum average cost per query</td>
        </tr>
        <tr>
            <td>❌ Failed Questions</td>
            <td>≤</td>
            <td>5</td>
            <td>Maximum number of individual question failures</td>
        </tr>
    </table>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Sidebar info
    with st.sidebar:
        st.markdown("### 🧪 LLM Eval Pipeline")
        st.markdown("---")
        st.markdown("**Quick Actions**")
        st.code("python main.py", language="bash")
        st.caption("Run the evaluation pipeline")
        st.markdown("---")
        st.markdown("**Tech Stack**")
        st.markdown("- 🤖 OpenAI GPT")
        st.markdown("- 📚 TF-IDF Retriever")
        st.markdown("- 📊 Plotly Charts")
        st.markdown("- 🔄 GitHub Actions CI/CD")
        st.markdown("---")
        st.caption("Built with Streamlit • v1.0.0")


if __name__ == "__main__":
    main()
