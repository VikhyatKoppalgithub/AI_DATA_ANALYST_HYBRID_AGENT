"""Streamlit front end.

    streamlit run app.py

Runs entirely locally against Ollama: no API key, no per-query cost. The layout
follows what the analysis actually guarantees — the narrative is presented as a
claim, and the contribution figures and verification checks that back it are
shown next to it rather than hidden behind an expander.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Set by the Hugging Face Space, which runs the 1.5b on 2 CPU cores. A visitor
# who does not know that reads slow, weaker answers as the project being bad,
# rather than as the documented cost of keeping the demo free and key-less.
DEMO_MODE = os.environ.get("ANALYST_DEMO") == "1"

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
DEMO_DATA = PROJECT_ROOT / "data" / "sales_2026.csv"
# The sandbox writes its snapshot and any charts here. Anchored to the project
# rather than the working directory: Streamlit's cwd depends on where it was
# launched, and a relative path silently splits "where charts are written" from
# "where the UI looks for them".
OUTPUTS = PROJECT_ROOT / "outputs"

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analyst.analysis import ChangeAnalysis
from analyst.llm import OllamaProvider, ProviderError
from analyst.llm.ollama import DEFAULT_MODEL
from analyst.prepare import prepare_frame
from analyst.profiler import load_and_profile, profile_dataframe
from analyst.session import AnalystSession

NEGATIVE = "#c0392b"
POSITIVE = "#1e8449"
MUTED = "#8899a6"

st.set_page_config(page_title="AI Data Analyst", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
      /* Surfaces are translucent greys rather than fixed colours: Streamlit
         follows the OS light/dark setting, and a hardcoded light background
         renders as white-on-white once the theme flips. */
      .block-container {max-width: 1200px; padding-top: 2.2rem;}
      [data-testid="stMetric"] {
        background: rgba(128,128,128,0.10);
        border: 1px solid rgba(128,128,128,0.22);
        padding: 14px 16px; border-radius: 12px;
      }
      .hero {padding:22px 26px;border-radius:16px;
        background:linear-gradient(115deg,#16202e,#2a3f6b);color:#fff;margin-bottom:22px;}
      .hero h1 {margin:0 0 6px 0;font-size:2rem;color:#fff;}
      .hero p {margin:0;color:#cdd8f0;font-size:0.95rem;}
      .answer {
        background: rgba(128,128,128,0.10);
        border-left: 4px solid #5b8def;
        color: inherit;
        padding: 16px 20px; border-radius: 8px;
        font-size: 1.05rem; line-height: 1.6;
      }
    </style>
    <div class="hero">
      <h1>AI Data Analyst</h1>
      <p>Upload messy data, ask a business question, get an answer whose every
         number is computed and verified — not generated.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------- sidebar


with st.sidebar:
    st.subheader("Model")
    model = st.text_input("Ollama model", value=DEFAULT_MODEL)
    provider = OllamaProvider(model)
    reachable, status = provider.health()
    (st.success if reachable else st.error)(status)
    if not reachable:
        st.code("brew services start ollama\nollama pull " + model, language="bash")

    st.caption(
        "The model only reads the question and writes the explanation. "
        "Every figure is computed in pandas and checked before it is shown."
    )
    st.divider()
    dedupe = st.checkbox("Remove duplicated rows", value=True)


# -------------------------------------------------------------- ingestion


if DEMO_MODE:
    st.info(
        "**Live demo — running `qwen2.5-coder:1.5b` on 2 CPU cores.** The "
        "published 98% eval score is the 14B on a Mac; this model scores 80% on "
        "the same suite, and mostly loses on routing. Expect a minute or more "
        "per question. Keeping the demo free and key-less is the tradeoff — the "
        "**Data profile** tab needs no model at all.",
        icon="🐌",
    )

uploaded = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xlsm", "parquet"])
demo = st.checkbox("Use the bundled demo dataset instead", value=not uploaded)

if not uploaded and not demo:
    st.info("Upload a file, or tick the demo dataset to try it.")
    st.stop()


@st.cache_data(show_spinner=False)
def load(payload: bytes | None, name: str, dedupe: bool):
    if payload is None:
        raw, profile = load_and_profile(DEMO_DATA)
    else:
        buffer = pd.io.common.BytesIO(payload)
        raw = pd.read_excel(buffer) if name.endswith((".xlsx", ".xlsm")) else pd.read_csv(buffer)
        profile = profile_dataframe(raw, source=name)
    return prepare_frame(raw, profile, drop_duplicates=dedupe)


try:
    data = load(
        uploaded.getvalue() if uploaded else None,
        uploaded.name if uploaded else "sales_2026.csv",
        dedupe,
    )
except Exception as exc:  # surfaced to the user; nothing to recover automatically
    st.error(f"Could not read that file: {exc}")
    st.stop()

a, b, c, d = st.columns(4)
a.metric("Rows", f"{data.rows_after:,}")
b.metric("Columns", len(data.frame.columns))
c.metric("Rows removed", f"{data.rows_removed:,}")
d.metric(
    "Data issues",
    len(data.quality_issues),
    help="Columns with missing values, duplicate rows, and mis-stored types.",
)


# ------------------------------------------------------------------ charts


def contribution_chart(analysis: ChangeAnalysis) -> go.Figure:
    """Sorted horizontal bars — the shape that reads best for contributions."""
    breakdown = analysis.intersection or next(iter(analysis.breakdowns.values()))
    # Rank in the direction of the overall change: on a rising metric, sorting
    # ascending would chart the largest counter-movements instead.
    slices = breakdown.ranked(analysis.direction)[:12][::-1]

    figure = go.Figure(
        go.Bar(
            x=[s.contribution_pp for s in slices],
            y=[s.segment for s in slices],
            orientation="h",
            marker_color=[NEGATIVE if s.contribution_pp < 0 else POSITIVE for s in slices],
            text=[f"{s.contribution_pp:+.2f} pp" for s in slices],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>contributes %{x:+.2f} pp<extra></extra>",
        )
    )
    figure.update_layout(
        title=f"Contribution to the {analysis.pct_change:+.1f}% change, by {breakdown.dimension}",
        xaxis_title="percentage points of the total change",
        yaxis_title=None,
        template="plotly_white",
        height=max(320, 32 * len(slices)),
        margin=dict(l=10, r=90, t=60, b=40),
    )
    return figure


def trend_chart(frame: pd.DataFrame, metric: str, date_column: str) -> go.Figure:
    series = (
        frame.dropna(subset=[date_column, metric])
        .groupby(frame[date_column].dt.to_period("M").astype(str))[metric]
        .sum()
    )
    figure = go.Figure(
        go.Scatter(x=list(series.index), y=list(series.values), mode="lines+markers")
    )
    figure.update_layout(
        title=f"{metric} by month",
        template="plotly_white",
        height=300,
        margin=dict(l=10, r=10, t=50, b=40),
    )
    return figure


# -------------------------------------------------------------------- tabs


ask_tab, profile_tab, data_tab = st.tabs(["Ask", "Data profile", "Prepared data"])

with profile_tab:
    st.subheader("What the analyst sees")
    st.caption(
        "The raw rows are never sent to the model — this profile is, which is why "
        "a 9,000-row file costs a few hundred tokens."
    )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "column": col.name,
                    "reads as": col.semantic_type,
                    "dtype": col.dtype,
                    "nulls": col.null_count,
                    "unique": col.unique_count,
                    "example": ", ".join(col.samples[:2]),
                }
                for col in data.profile.columns
            ]
        ),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Cleaning log")
    for action in data.actions:
        icon = "⚠️" if not action.reversible else "✓"
        st.write(f"{icon} {action.description}")

    if data.quality_issues:
        st.subheader("Data issues")
        for issue in data.quality_issues:
            st.write(f"• {issue}")

with data_tab:
    st.dataframe(data.frame.head(1000), width="stretch")
    st.download_button(
        "Download prepared CSV",
        data.frame.to_csv(index=False).encode(),
        "prepared_data.csv",
        "text/csv",
    )

with ask_tab:
    question = st.text_input(
        "What do you want to understand?",
        placeholder="Please enter your question — e.g. Why did revenue drop in March?",
    )
    # Empty input previously left the button live but did nothing on click.
    run = st.button(
        "Analyse", type="primary", disabled=not reachable or not question.strip()
    )

    if not reachable:
        st.warning("Start Ollama to ask questions. The other tabs work without it.")

    if run and question.strip():
        # One session per (dataset, model), kept in session_state. Building a new
        # one per query would spawn a sandbox worker per question and leak it —
        # st.stop() below means no teardown runs at the end of a script pass.
        key = (data.profile.source, data.rows_after, model)
        if st.session_state.get("session_key") != key:
            previous = st.session_state.get("session")
            if previous is not None:
                previous.close()
            st.session_state["session"] = AnalystSession(provider, data, workdir=OUTPUTS)
            st.session_state["session_key"] = key
        session = st.session_state["session"]

        try:
            with st.spinner("Routing, computing, then explaining…"):
                answer = session.ask(question)
        except ProviderError as exc:
            st.error(str(exc))
            st.stop()
        except (ValueError, RuntimeError) as exc:
            st.error(f"Could not answer that: {exc}")
            st.stop()

        st.markdown(f'<div class="answer">{answer.narrative}</div>', unsafe_allow_html=True)

        # ---------------------------------------------------- generated-code route
        if answer.route == "code":
            st.warning(
                "**Unverified.** This question needed generated code, so the "
                "figures come from Python the model wrote and ran — they are not "
                "reconciled the way the change-analysis engine's are. The code "
                "ran successfully and its output is real, but nothing checked "
                "that it computed the right thing. Read the code below before "
                "relying on the numbers.",
                icon="⚠️",
            )
            if answer.code.stop_reason == "no_execution":
                st.error(
                    "The model tried to answer without running any code, so its "
                    "figures would have been invented. The answer was refused."
                )

            for index, attempt in enumerate(answer.code.attempts, 1):
                state = "ok" if attempt.ok else "failed"
                with st.expander(
                    f"Step {index} of {len(answer.code.attempts)} — {state}"
                    f" ({attempt.result.duration_ms} ms)",
                    expanded=len(answer.code.attempts) == 1,
                ):
                    st.code(attempt.code, language="python")
                    st.code(attempt.result.to_model_text()[:3000], language="text")

            for chart in answer.code.charts:
                path = OUTPUTS / chart
                if path.exists():
                    st.image(str(path))

            st.caption(
                f"route: generated code · {len(answer.completions)} local model calls · "
                f"{answer.output_tokens:,} output tokens · {answer.seconds:.1f}s · $0.00"
            )
            st.stop()

        # ------------------------------------------------ verified change route
        analysis = answer.change.analysis
        st.success("**Verified.** Every figure below is computed and reconciled.", icon="✅")

        left, right = st.columns([3, 2])

        with left:
            st.plotly_chart(contribution_chart(analysis), width="stretch")
            st.plotly_chart(
                trend_chart(data.frame, analysis.metric, analysis.date_column),
                width="stretch",
            )

        with right:
            st.subheader("Verification")
            for check in analysis.checks:
                st.write(f"{'✅' if check.passed else '❌'} {check.name}")
                st.caption(check.detail)

            decoys = [
                s
                for s in (
                    analysis.intersection.ranked(analysis.direction)
                    if analysis.intersection
                    else []
                )
                if s.is_decoy
            ]
            if decoys:
                st.subheader("Large moves that explain little")
                for s in decoys[:3]:
                    st.write(
                        f"**{s.segment}** moved {s.own_pct_change:+.1f}% but explains "
                        f"only {s.contribution_pp:+.2f} pp — it is "
                        f"{s.share_of_total_before:.1f}% of the total."
                    )

            st.subheader("How this was read")
            st.write(f"Metric: `{analysis.metric}` · Date: `{analysis.date_column}`")
            st.write(f"Compared **{analysis.period_before}** with **{analysis.period_after}**")
            for warning in answer.change.plan.warnings:
                st.warning(warning)

        with st.expander("Every dimension, separately"):
            st.caption(
                "Each block below partitions the *same* change a different way. "
                "They are alternative views, not additive causes — do not sum across them."
            )
            for name, breakdown in analysis.breakdowns.items():
                flag = "" if breakdown.reconciles else "  ⚠️ does not reconcile"
                st.markdown(f"**{name}**{flag}")
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "segment": s.segment,
                                "before": round(s.before, 2),
                                "after": round(s.after, 2),
                                "change": round(s.change, 2),
                                "contribution (pp)": round(s.contribution_pp, 2),
                                "own change (%)": round(s.own_pct_change, 1),
                            }
                            for s in breakdown.slices
                        ]
                    ),
                    width="stretch",
                    hide_index=True,
                )

        if analysis.notes:
            with st.expander("Normalisation applied before grouping"):
                for note in analysis.notes:
                    st.write(f"• {note}")

        st.caption(
            f"route: verified change analysis · {len(answer.completions)} local model "
            f"calls · {answer.output_tokens:,} output tokens · {answer.seconds:.1f}s · $0.00"
        )
