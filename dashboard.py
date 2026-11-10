"""
Predictd NBA Streamlit dashboard.

Run:
  streamlit run dashboard.py

Tabs:
  1. Today's Picks      (predictions/latest.csv)
  2. History            (predictions/tracked.csv)
  3. Accuracy           (same, with charts)
  4. Monitor            (latest logs/monitor_*.log)
  5. Bet Slips          (predictions/bet_slips_*.csv + logs/bet_slip_accuracy.csv)
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --- [SECRETS HANDLING] ---
# Try to load from Streamlit Cloud secrets first, fall back to local .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def get_api_key():
    """Get API key from Streamlit secrets (cloud) or environment (local)."""
    if "ODDS_API_KEY" in st.secrets:
        return st.secrets["ODDS_API_KEY"]
    return os.getenv("ODDS_API_KEY", "")

# Store it so other modules can access it if needed
API_KEY = get_api_key()

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
PRED_DIR = ROOT / "predictions"
LOGS_DIR = ROOT / "logs"

LATEST_CSV = PRED_DIR / "latest.csv"
TRACKED_CSV = PRED_DIR / "tracked.csv"
BET_SLIP_ACC_CSV = LOGS_DIR / "bet_slip_accuracy.csv"

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Predictd NBA",
    page_icon="🏀",
    layout="wide",
)

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def file_status(path: Path) -> tuple:
    if not path.exists():
        return "🔴", "missing"
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return "🟢", mtime.strftime("%Y-%m-%d %H:%M")

def run_pipeline_step(script_name: str) -> tuple:
    script_path = SRC / script_name
    if not script_path.exists():
        return False, f"Missing script: {script_path}"
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True, text=True, timeout=180,
        )
        if result.returncode == 0:
            return True, result.stdout[-1500:] if result.stdout else "(no output)"
        return False, result.stderr[-2000:] if result.stderr else "(no error text)"
    except subprocess.TimeoutExpired:
        return False, f"Timed out after 180s running {script_name}"

def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Failed to read {path.name}: {e}")
        return pd.DataFrame()

def to_utc_string(series: pd.Series, fmt="%a %b %d %H:%M UTC") -> pd.Series:
    try:
        return pd.to_datetime(series, utc=True).dt.strftime(fmt)
    except Exception:
        return series

def style_pick(val: str) -> str:
    if val in ("HOME", "OVER"):
        return "background-color:#1f6f3f; color:white; font-weight:600"
    if val in ("AWAY", "UNDER"):
        return "background-color:#8a2a2a; color:white; font-weight:600"
    return "background-color:#333; color:#999"

def style_hit(val):
    try:
        v = float(val)
    except (TypeError, ValueError):
        return ""
    if pd.isna(v):
        return ""
    if v == 1:
        return "background-color:#1f6f3f; color:white; font-weight:600"
    if v == 0:
        return "background-color:#8a2a2a; color:white; font-weight:600"
    return ""

def find_latest_bet_slip_file() -> Path | None:
    files = sorted(glob.glob(str(PRED_DIR / "bet_slips_*.csv")))
    return Path(files[-1]) if files else None

def bet_slip_status_dot(combined_prob: float) -> str:
    if combined_prob >= 0.5:
        return "🟢"
    if combined_prob >= 0.25:
        return "🟡"
    return "🔴"

STRATEGY_LABELS = {
    "top_mixed": "🏆 Best overall picks across ML/Spread/Total",
    "ml_best": "💰 Best Moneyline picks only",
    "spread_best": "📊 Best Spread picks only",
    "total_best": "📈 Best Over/Under picks only",
}

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.title("🏀 Predictd NBA")
st.caption("Moneyline · Spread · Total — NBA prediction dashboard")

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("System status")

    latest_bs = find_latest_bet_slip_file()
    for label, path in [
        ("latest predictions", LATEST_CSV),
        ("tracked results", TRACKED_CSV),
        ("latest bet slips", latest_bs if latest_bs else PRED_DIR / "__none__"),
        ("bet slip accuracy", BET_SLIP_ACC_CSV),
        ("monitor log", LOGS_DIR / f"monitor_{datetime.now().strftime('%Y-%m-%d')}.log"),
    ]:
        emoji, when = file_status(path)
        st.markdown(f"{emoji} **{label}**  \n<small>{when}</small>",
                    unsafe_allow_html=True)

    st.divider()
    st.header("Actions")

    if st.button("🔄 Refresh odds + predictions + slips", use_container_width=True):
        with st.spinner("Fetching odds, predicting, and building bet slips ..."):
            ok1, out1 = run_pipeline_step("fetch_odds.py")
            ok2, out2 = run_pipeline_step("predict_live.py")
            ok3, out3 = run_pipeline_step("bet_slip_builder.py")
        if ok1 and ok2 and ok3:
            st.success("Odds refreshed, predictions and slips regenerated.")
            st.rerun()
        else:
            st.error("Refresh failed.")
            with st.expander("fetch_odds.py"):
                st.code(out1)
            with st.expander("predict_live.py"):
                st.code(out2)
            with st.expander("bet_slip_builder.py"):
                st.code(out3)

    if st.button("📊 Update results + monitor", use_container_width=True):
        with st.spinner("Checking finished games and updating monitor ..."):
            ok1, out1 = run_pipeline_step("track_accuracy.py")
            ok2, out2 = run_pipeline_step("adaptive_monitor.py")
        if ok1 and ok2:
            st.success("Results updated.")
            st.rerun()
        else:
            st.error("Update failed.")
            with st.expander("track_accuracy.py"):
                st.code(out1)
            with st.expander("adaptive_monitor.py"):
                st.code(out2)

    st.divider()
    st.caption(f"Project: `{ROOT.name}`  \nBuilt: {datetime.now():%Y-%m-%d}")

# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------
tab_picks, tab_history, tab_accuracy, tab_betslips, tab_monitor = st.tabs(
    ["📅 Today's Picks", "📜 History", "📈 Accuracy", "🎫 Bet Slips", "🩺 Monitor"]
)

# ===========================================================================
# TAB 1 — TODAY'S PICKS
# ===========================================================================
with tab_picks:
    st.subheader("Upcoming predictions")

    latest = safe_read_csv(LATEST_CSV)
    if latest.empty:
        st.info("No predictions yet. Click **Refresh odds + predictions + slips** in the sidebar.")
    else:
        latest = latest.copy()
        latest["commence_time"] = pd.to_datetime(latest["commence_time"], utc=True)
        latest = latest.sort_values("commence_time").reset_index(drop=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            ml_filter = st.multiselect(
                "Moneyline", ["HOME", "AWAY", "PASS"],
                default=["HOME", "AWAY"], key="ml_filter",
            )
        with c2:
            spread_filter = st.multiselect(
                "Spread", ["HOME", "AWAY", "PASS"],
                default=["HOME", "AWAY"], key="sp_filter",
            )
        with c3:
            total_filter = st.multiselect(
                "Total", ["OVER", "UNDER", "PASS"],
                default=["OVER", "UNDER"], key="tot_filter",
            )

        mask = (
            latest["ml_pick"].isin(ml_filter) |
            latest["spread_pick"].isin(spread_filter) |
            latest["total_pick"].isin(total_filter)
        )
        view = latest[mask].copy()

        if view.empty:
            st.warning("No games match the selected filters.")
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Games", len(view))
            m2.metric("ML picks", int((view["ml_pick"].isin(["HOME", "AWAY"])).sum()))
            m3.metric("Spread picks", int((view["spread_pick"].isin(["HOME", "AWAY"])).sum()))
            m4.metric("Total picks", int((view["total_pick"].isin(["OVER", "UNDER"])).sum()))

            st.divider()

            for row in view.itertuples(index=False):
                ct = row.commence_time.strftime("%a %b %d · %H:%M UTC")
                header = f"{ct}  —  **{row.away_team}** @ **{row.home_team}**"
                with st.expander(header, expanded=False):
                    st.caption(f"Book: `{row.bookmaker}`  ·  Season: `{row.season}`")

                    c1, c2, c3 = st.columns(3)

                    with c1:
                        st.markdown("**Moneyline**")
                        if not pd.isna(row.market_home_prob):
                            st.markdown(
                                f"Model: **{row.model_home_win_prob*100:.1f}%** home / "
                                f"{row.model_away_win_prob*100:.1f}% away"
                            )
                            st.markdown(
                                f"Market: {row.market_home_prob*100:.1f}% / "
                                f"{row.market_away_prob*100:.1f}%"
                            )
                        if row.ml_pick != "PASS":
                            st.markdown(
                                f"**Pick: :{('green' if row.ml_pick=='HOME' else 'red')}[{row.ml_pick}]**"
                                f"  ·  edge **{row.ml_edge*100:+.1f}%**"
                            )
                        else:
                            st.markdown("*No edge — PASS*")

                    with c2:
                        st.markdown("**Spread**")
                        st.markdown(f"Model margin: **{row.model_margin:+.1f}**")
                        st.markdown(f"Market line: **{row.market_spread:+.1f}**")
                        if row.spread_pick != "PASS":
                            st.markdown(
                                f"**Pick: {row.spread_pick}**  ·  "
                                f"edge **{row.spread_edge:+.1f}** pts"
                            )
                        else:
                            st.markdown("*No edge — PASS*")

                    with c3:
                        st.markdown("**Total**")
                        st.markdown(f"Model total: **{row.model_total:.1f}**")
                        st.markdown(f"Market line: **{row.market_total:.1f}**")
                        if row.total_pick != "PASS":
                            st.markdown(
                                f"**Pick: {row.total_pick}**  ·  "
                                f"edge **{row.total_edge:+.1f}** pts"
                            )
                        else:
                            st.markdown("*No edge — PASS*")

# ===========================================================================
# TAB 2 — HISTORY
# ===========================================================================
with tab_history:
    st.subheader("Resolved predictions")

    tracked = safe_read_csv(TRACKED_CSV)
    if tracked.empty:
        st.info("No resolved games yet. Click **Update results + monitor** once games have finished.")
    else:
        tracked = tracked.copy()
        if "commence_time" in tracked.columns:
            tracked["commence_time"] = pd.to_datetime(tracked["commence_time"], utc=True)
            tracked = tracked.sort_values("commence_time", ascending=False)

        show_cols = [
            "commence_time", "home_team", "away_team",
            "ml_pick", "ml_correct", "ml_edge",
            "spread_pick", "spread_correct",
            "total_pick", "total_correct",
            "actual_home_pts", "actual_away_pts",
        ]
        show_cols = [c for c in show_cols if c in tracked.columns]
        v = tracked[show_cols].copy()
        if "commence_time" in v.columns:
            v["commence_time"] = to_utc_string(v["commence_time"], "%a %b %d %H:%M")

        styled = (
            v.style
            .map(style_pick, subset=[c for c in ["ml_pick", "spread_pick", "total_pick"] if c in v.columns])
            .map(style_hit, subset=[c for c in ["ml_correct", "spread_correct", "total_correct"] if c in v.columns])
            .format(precision=2, na_rep="—")
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

        st.caption(f"Total resolved: **{len(v)}**  ·  "
                   f"ML wins: **{int(tracked['ml_correct'].sum()) if 'ml_correct' in tracked else 0}**")

# ===========================================================================
# TAB 3 — ACCURACY
# ===========================================================================
with tab_accuracy:
    st.subheader("Accuracy report")

    tracked = safe_read_csv(TRACKED_CSV)
    if tracked.empty:
        st.info("Not enough resolved games yet.")
    else:
        tracked = tracked.copy()
        if "commence_time" in tracked.columns:
            tracked["commence_time"] = pd.to_datetime(tracked["commence_time"], utc=True)
            tracked = tracked.sort_values("commence_time").reset_index(drop=True)

        def acc_of(df, pick_col, hit_col):
            r = df[df[pick_col].isin(["HOME", "AWAY", "OVER", "UNDER"])]
            if r.empty:
                return None
            return float(r[hit_col].mean()), len(r)

        ml_acc = acc_of(tracked, "ml_pick", "ml_correct")
        sp_acc = acc_of(tracked, "spread_pick", "spread_correct")
        tot_acc = acc_of(tracked, "total_pick", "total_correct")

        c1, c2, c3 = st.columns(3)
        for col, label, val in [
            (c1, "Moneyline", ml_acc),
            (c2, "Spread", sp_acc),
            (c3, "Total", tot_acc),
        ]:
            with col:
                if val is None:
                    st.metric(label, "—", "no picks")
                else:
                    acc, n = val
                    st.metric(label, f"{acc*100:.1f}%", f"{n} picks")

        st.divider()
        st.markdown("### Rolling accuracy (last 30 games)")

        fig = go.Figure()
        for label, pick_col, hit_col, color in [
            ("Moneyline", "ml_pick", "ml_correct", "#4CAF50"),
            ("Spread", "spread_pick", "spread_correct", "#FFC107"),
            ("Total", "total_pick", "total_correct", "#2196F3"),
        ]:
            sub = tracked[tracked[pick_col].isin(["HOME", "AWAY", "OVER", "UNDER"])].copy()
            if sub.empty or len(sub) < 5:
                continue
            sub["rolling"] = sub[hit_col].rolling(30, min_periods=5).mean()
            fig.add_trace(go.Scatter(
                x=sub["commence_time"], y=sub["rolling"],
                mode="lines", name=label, line=dict(color=color, width=2),
            ))
        fig.add_hline(y=0.5, line_dash="dot", line_color="gray")
        fig.update_layout(
            yaxis_title="accuracy", yaxis_range=[0, 1],
            height=340, margin=dict(l=0, r=0, t=20, b=0),
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.markdown("### Calibration (moneyline)")

        ml = tracked[tracked["ml_pick"].isin(["HOME", "AWAY"])].copy()
        if not ml.empty:
            def conf(row):
                return row["model_home_win_prob"] if row["ml_pick"] == "HOME" \
                    else 1 - row["model_home_win_prob"]
            ml["confidence"] = ml.apply(conf, axis=1)
            bins = [0.5, 0.55, 0.60, 0.65, 0.70, 0.80, 1.00]
            labels = ["0.50-0.55", "0.55-0.60", "0.60-0.65",
                      "0.65-0.70", "0.70-0.80", "0.80-1.00"]
            ml["bucket"] = pd.cut(ml["confidence"], bins=bins, labels=labels, right=False)

            cal = ml.groupby("bucket", observed=True).agg(
                n=("ml_correct", "size"),
                expected=("confidence", "mean"),
                actual=("ml_correct", "mean"),
            ).reset_index()
            cal["gap"] = cal["actual"] - cal["expected"]

            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=cal["bucket"].astype(str), y=cal["expected"],
                name="expected", marker_color="#888",
            ))
            fig2.add_trace(go.Bar(
                x=cal["bucket"].astype(str), y=cal["actual"],
                name="actual", marker_color="#4CAF50",
            ))
            fig2.update_layout(
                barmode="group", yaxis_title="accuracy",
                height=300, margin=dict(l=0, r=0, t=20, b=0),
                legend=dict(orientation="h", y=1.1),
            )
            st.plotly_chart(fig2, use_container_width=True)
            st.dataframe(cal.style.format({"expected": "{:.3f}", "actual": "{:.3f}",
                                          "gap": "{:+.3f}"}, na_rep="—"),
                         use_container_width=True, hide_index=True)

        if "bookmaker" in tracked.columns:
            st.divider()
            st.markdown("### Moneyline accuracy by bookmaker")
            bk = (tracked[tracked["ml_pick"].isin(["HOME", "AWAY"])]
                  .groupby("bookmaker")
                  .agg(n=("ml_correct", "size"), accuracy=("ml_correct", "mean"))
                  .reset_index().sort_values("n", ascending=False))
            if not bk.empty:
                fig3 = px.bar(bk, x="bookmaker", y="accuracy",
                              text=bk["accuracy"].map(lambda v: f"{v:.2f}"),
                              color="accuracy", color_continuous_scale="RdYlGn",
                              range_color=[0.4, 0.8])
                fig3.update_layout(yaxis_range=[0, 1], height=280,
                                   margin=dict(l=0, r=0, t=20, b=0),
                                   coloraxis_showscale=False)
                st.plotly_chart(fig3, use_container_width=True)

# ===========================================================================
# TAB 4 — BET SLIPS (NEW)
# ===========================================================================
with tab_betslips:
    st.subheader("🎫 Bet Slips")

    bs_path = find_latest_bet_slip_file()

    if bs_path is None:
        st.info("No bet slip files yet. Click **Refresh odds + predictions + slips** in the sidebar.")
    else:
        try:
            bs = pd.read_csv(bs_path)
        except Exception as e:
            st.error(f"Failed to read {bs_path.name}: {e}")
            bs = pd.DataFrame()

        if bs.empty:
            st.warning(f"{bs_path.name} is empty.")
        else:
            c1, c2 = st.columns([3, 3])
            all_strats = sorted(bs["strategy"].unique().tolist())
            all_markets = sorted(bs["market"].unique().tolist())

            with c1:
                strat_filter = st.multiselect(
                    "Strategy", all_strats, default=all_strats,
                    key="bs_strat_filter",
                )
            with c2:
                market_filter = st.multiselect(
                    "Market", all_markets, default=all_markets,
                    key="bs_mkt_filter",
                )

            filtered = bs[
                bs["strategy"].isin(strat_filter) & bs["market"].isin(market_filter)
            ].copy()

            n_slips = filtered.groupby(["strategy", "slip_id"]).ngroups

            st.caption(f"📄 Showing **{n_slips} slips** from `{bs_path.name}`")
            st.divider()

            if n_slips == 0:
                st.warning("No slips match the selected filters.")
            else:
                for strat in all_strats:
                    if strat not in strat_filter:
                        continue
                    strat_df = filtered[filtered["strategy"] == strat]
                    if strat_df.empty:
                        continue

                    st.markdown(f"#### {STRATEGY_LABELS.get(strat, strat)}")
                    st.caption(
                        f"{strat_df['slip_id'].nunique()} slips · "
                        f"markets: {sorted(strat_df['market'].unique().tolist())}"
                    )

                    for slip_id, slip_df in strat_df.groupby("slip_id"):
                        slip_df = slip_df.sort_values("pick_num").reset_index(drop=True)
                        size = int(slip_df["size"].iloc[0])
                        combined = float(slip_df["combined_prob"].iloc[0])
                        n_games = slip_df["HomeTeam"].nunique()
                        odds = 1.0 / combined if combined > 0 else 0
                        dot = bet_slip_status_dot(combined)

                        header = (
                            f"{dot} Slip #{int(slip_id)} — {size}-fold | "
                            f"Combined: {combined*100:.1f}% | "
                            f"Odds: {odds:.2f}x | Games: {n_games}"
                        )

                        with st.expander(header, expanded=False):
                            display = slip_df[[
                                "pick_num", "Date", "HomeTeam", "AwayTeam",
                                "Market_Type", "pick_value", "probability", "score",
                            ]].rename(columns={
                                "pick_num": "#",
                                "HomeTeam": "Home",
                                "AwayTeam": "Away",
                                "Market_Type": "Market",
                                "pick_value": "Pick",
                                "probability": "Prob",
                                "score": "Score",
                            }).copy()

                            display["Match"] = display["Away"] + " @ " + display["Home"]
                            display = display[[
                                "#", "Date", "Match", "Market", "Pick", "Prob", "Score"
                            ]]

                            styled = display.style.format({
                                "Prob": "{:.3f}",
                                "Score": "{:.3f}",
                            }, na_rep="—")
                            st.dataframe(styled, use_container_width=True, hide_index=True)

                    st.divider()

    st.subheader("📊 Historical Bet Slip Accuracy")

    if not BET_SLIP_ACC_CSV.exists():
        st.info(
            "No fully-played slips yet. Results appear once all picks in a slip resolve."
        )
    else:
        try:
            acc = pd.read_csv(BET_SLIP_ACC_CSV)
        except Exception as e:
            st.error(f"Failed to read {BET_SLIP_ACC_CSV.name}: {e}")
            acc = pd.DataFrame()

        if acc.empty:
            st.info("Bet slip accuracy file is empty.")
        else:
            n = len(acc)
            wins = int((acc["outcome"] == "WIN").sum())
            win_rate = wins / n if n else 0.0
            avg_pred = float(acc["combined_prob"].mean())

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Slips Evaluated", n)
            m2.metric("Wins", wins)
            m3.metric("Win Rate", f"{win_rate*100:.1f}%")
            m4.metric("Avg Predicted", f"{avg_pred*100:.1f}%")

            gap = avg_pred - win_rate
            if gap > 0.05:
                st.error(
                    f"⚠ Overconfident by {gap*100:.1f} pp "
                    f"(predicted {avg_pred*100:.1f}%, actual {win_rate*100:.1f}%)"
                )
            elif gap < -0.05:
                st.success(
                    f"✓ Under-confident by {-gap*100:.1f} pp (good) "
                    f"(predicted {avg_pred*100:.1f}%, actual {win_rate*100:.1f}%)"
                )
            else:
                st.success(
                    f"✓ Well-calibrated (gap {gap*100:+.1f} pp)"
                )

            st.divider()

            st.markdown("### Win rate by strategy")
            by_strat = (acc.groupby("strategy")
                        .agg(n=("outcome", "size"),
                             wins=("outcome", lambda s: (s == "WIN").sum()))
                        .reset_index())
            by_strat["win_rate"] = by_strat["wins"] / by_strat["n"]

            if not by_strat.empty:
                fig_s = px.bar(
                    by_strat, x="strategy", y="win_rate",
                    text=by_strat["win_rate"].map(lambda v: f"{v*100:.0f}%"),
                    color="win_rate", color_continuous_scale="RdYlGn",
                    range_color=[0.0, 1.0],
                )
                fig_s.add_hline(y=win_rate, line_dash="dot", line_color="gray")
                fig_s.update_layout(
                    yaxis_range=[0, 1], height=300,
                    margin=dict(l=0, r=0, t=20, b=0),
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig_s, use_container_width=True)

            st.markdown("### Win rate by slip size")
            by_size = (acc.groupby("size")
                       .agg(n=("outcome", "size"),
                            wins=("outcome", lambda s: (s == "WIN").sum()))
                       .reset_index())
            by_size["win_rate"] = by_size["wins"] / by_size["n"]
            by_size["size_label"] = by_size["size"].astype(str) + "-fold"

            if not by_size.empty:
                fig_z = px.bar(
                    by_size, x="size_label", y="win_rate",
                    text=by_size["win_rate"].map(lambda v: f"{v*100:.0f}%"),
                    color="win_rate", color_continuous_scale="RdYlGn",
                    range_color=[0.0, 1.0],
                )
                fig_z.add_hline(y=win_rate, line_dash="dot", line_color="gray")
                fig_z.update_layout(
                    yaxis_range=[0, 1], height=300,
                    margin=dict(l=0, r=0, t=20, b=0),
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig_z, use_container_width=True)

            st.markdown("### All evaluated slips")
            tbl = acc.sort_values(
                ["slip_date", "strategy", "size"], ascending=[False, True, True]
            ).reset_index(drop=True)

            def style_outcome(v):
                if v == "WIN":
                    return "background-color:#1f6f3f; color:white; font-weight:600"
                if v == "LOSS":
                    return "background-color:#8a2a2a; color:white; font-weight:600"
                return ""

            styled_tbl = tbl.style.map(style_outcome, subset=["outcome"]).format(
                {"combined_prob": "{:.3f}"}, na_rep="—"
            )
            st.dataframe(styled_tbl, use_container_width=True, hide_index=True)

# ===========================================================================
# TAB 5 — MONITOR
# ===========================================================================
with tab_monitor:
    st.subheader("Adaptive monitor")

    log_files = sorted(LOGS_DIR.glob("monitor_*.log"), reverse=True)
    if not log_files:
        st.info("No monitor logs yet. Click **Update results + monitor** in the sidebar.")
    else:
        chosen = st.selectbox(
            "Log file", [p.name for p in log_files],
            index=0, key="log_pick",
        )
        chosen_path = LOGS_DIR / chosen
        content = chosen_path.read_text(encoding="utf-8", errors="replace")

        alerts = [ln for ln in content.splitlines() if "[!]" in ln]

        if alerts:
            st.error(f"**{len(alerts)} alert(s) in this log**")
            for a in alerts:
                st.markdown(f"- {a.strip()}")
        else:
            st.success("No alerts — everything within expected range.")

        with st.expander("Full log", expanded=True):
            st.code(content, language="text")

# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    f"Predictd NBA · built {datetime.now():%Y-%m-%d} · "
    f"project: `{ROOT}`"
)