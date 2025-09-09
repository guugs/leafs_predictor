import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import altair as alt


#constants
TEAM = "TOR"
APP_DIR = Path(__file__).resolve().parent
LOGO_PATH = APP_DIR.parent / "images" / "logo.png"

st.set_page_config(page_title="Leafs Elo Predictor",
                   page_icon=str(LOGO_PATH),
                   layout="wide")


NAME_TO_ABBR = {
    "Anaheim Ducks": "ANA","Arizona Coyotes": "ARI","Boston Bruins": "BOS","Buffalo Sabres": "BUF",
    "Calgary Flames": "CGY","Carolina Hurricanes": "CAR","Chicago Blackhawks": "CHI","Colorado Avalanche": "COL",
    "Columbus Blue Jackets": "CBJ","Dallas Stars": "DAL","Detroit Red Wings": "DET","Edmonton Oilers": "EDM",
    "Florida Panthers": "FLA","Los Angeles Kings": "LAK","Minnesota Wild": "MIN","Montreal Canadiens": "MTL",
    "Nashville Predators": "NSH","New Jersey Devils": "NJD","New York Islanders": "NYI","New York Rangers": "NYR",
    "Ottawa Senators": "OTT","Philadelphia Flyers": "PHI","Pittsburgh Penguins": "PIT","San Jose Sharks": "SJS",
    "Seattle Kraken": "SEA","St. Louis Blues": "STL","Tampa Bay Lightning": "TBL","Toronto Maple Leafs": "TOR",
    "Utah Hockey Club": "UTA","Vancouver Canucks": "VAN","Vegas Golden Knights": "VGK","Washington Capitals": "WSH",
    "Winnipeg Jets": "WPG"
}
#default elo settings
HOME_EDGE = 5.0          
B2B_PENALTY = 10.0        
REST_PTS_PER_DAY = 3.0    
ELO_SCALE_DEN = 600.0     # larger => flatter probabilities
ALPHA_ELO = 0.65          # 1.0 => pure Elo, 0.0 => coin flip


def allocate_otl(loss_idx: np.ndarray, probs: np.ndarray, expected_share: float) -> set:
    """Pick the subset of predicted losses to mark as OTL, matching expected_share, by closeness to 0.5."""
    n = len(loss_idx)
    k = int(round(max(0.0, min(1.0, expected_share)) * n))
    if n == 0 or k == 0:
        return set()
    closeness = np.abs(probs[loss_idx] - 0.5)
    otl_pick = loss_idx[np.argsort(closeness)[:k]]
    return set(otl_pick)

@st.cache_data(show_spinner=False)
def load_leafs_schedule_from_csv(csv_path: str, team_abbr: str = TEAM,
                                 date_col: str = "Date", home_col: str = "Home Team",
                                 away_col: str = "Away Team") -> pd.DataFrame:
    """Load schedule CSV and compute rest features (Leafs POV)."""
    df = pd.read_csv(csv_path)
    for col in (date_col, home_col, away_col):
        if col not in df.columns:
            raise KeyError(f"Missing column '{col}'. Found: {list(df.columns)}")
    df = df.rename(columns={date_col: "date", home_col: "home_team", away_col: "away_team"})

    # Robust date parsing: DD/MM/YYYY preferred, ISO fallback
    s = df["date"].astype(str).str.strip()
    parsed = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    na = parsed.isna()
    if na.any():
        parsed.loc[na] = pd.to_datetime(s[na], errors="coerce")
    df["date"] = parsed

    # Leafs perspective
    df["home"] = (df["home_team"] == team_abbr).astype(int)
    df["opponent"] = np.where(df["home"] == 1, df["away_team"], df["home_team"])

    # Vectorized rest features
    df = df.sort_values("date").reset_index(drop=True)
    d_days = df["date"].diff().dt.days.fillna(2).clip(lower=0).astype(int)
    df["rest_days"] = d_days
    df["back_to_back"] = (d_days == 1).astype(int)

    # Placeholder until opponent rest is computed
    df["rest_diff"] = 0

    # Keep nice order; pass through helpful refs if present
    order = ["date","home_team","away_team","opponent","home","back_to_back","rest_days","rest_diff"]
    for opt in ["Location","Result","Match Number","Round Number"]:
        if opt in df.columns: order.append(opt)
    return df[order]

@st.cache_data(show_spinner=False)
def build_elo_map(teams_csv: str, save_pct_csv: str) -> tuple[dict, pd.DataFrame]:
    """Build composite team strength (EV/PP/PK + Save%) and map to Elo."""
    teams_df = pd.read_csv(teams_csv).copy()
    teams_df["season"] = pd.to_numeric(teams_df["season"], errors="coerce")
    latest = int(teams_df["season"].max())

    t = teams_df.loc[teams_df["season"] == latest, ["team","situation","iceTime","xGoalsFor","xGoalsAgainst"]].copy()
    SIT_MAP = {"5on5":"EV", "5on4":"PP", "4on5":"PK"}
    t["SIT"] = t["situation"].map(SIT_MAP)

    t["iceTime"] = pd.to_numeric(t["iceTime"], errors="coerce").replace(0, np.nan)
    t["xGoalsFor"] = pd.to_numeric(t["xGoalsFor"], errors="coerce")
    t["xGoalsAgainst"] = pd.to_numeric(t["xGoalsAgainst"], errors="coerce")
    t["xGF60"] = (t["xGoalsFor"] / t["iceTime"]) * 60.0
    t["xGA60"] = (t["xGoalsAgainst"] / t["iceTime"]) * 60.0
    t["net_xG60"] = t["xGF60"] - t["xGA60"]

    pp = t.loc[t["SIT"]=="PP"].groupby("team", as_index=False)[["xGF60"]].mean()
    pk = t.loc[t["SIT"]=="PK"].groupby("team", as_index=False)[["xGA60"]].mean()
    ev = t.loc[t["SIT"]=="EV"].groupby("team", as_index=False)[["net_xG60"]].mean()

    all_teams = pd.DataFrame({"team": t["team"].dropna().unique()})
    pp = all_teams.merge(pp, on="team", how="left"); pp["xGF60"] = pp["xGF60"].fillna(pp["xGF60"].mean())
    pk = all_teams.merge(pk, on="team", how="left"); pk["xGA60"] = pk["xGA60"].fillna(pk["xGA60"].mean())
    ev = all_teams.merge(ev, on="team", how="left"); ev["net_xG60"] = ev["net_xG60"].fillna(ev["net_xG60"].mean())

    def z(s: pd.Series) -> pd.Series:
        s = s.astype(float)
        mu = s.mean(); sd = s.std(ddof=0)
        return (s - mu) / (sd if sd != 0 else 1.0)

    ev_z = z(ev["net_xG60"])
    pp_z = z(pp["xGF60"])
    pk_z = z(-pk["xGA60"])  # lower xGA60 is better

    sv_df = pd.read_csv(save_pct_csv).rename(columns={"team":"team_name","savePct":"savePct"})
    sv_df["team"] = sv_df["team_name"].map(NAME_TO_ABBR)
    sv = all_teams.merge(sv_df[["team","savePct"]], on="team", how="left")
    sv["savePct"] = pd.to_numeric(sv["savePct"], errors="coerce")
    sv["savePct"] = sv["savePct"].fillna(sv["savePct"].mean())
    sv_z = z(sv["savePct"])

    W_EV, W_PP, W_PK, W_SV = 0.60, 0.20, 0.15, 0.05
    comp = all_teams.copy()
    comp["ev_z"] = ev_z.values
    comp["pp_z"] = pp_z.values
    comp["pk_z"] = pk_z.values
    comp["sv_z"] = sv_z.values
    comp["z_composite"] = W_EV*comp["ev_z"] + W_PP*comp["pp_z"] + W_PK*comp["pk_z"] + W_SV*comp["sv_z"]
    comp["elo"] = 1500.0 + 100.0 * comp["z_composite"]

    elo_map = dict(zip(comp["team"], comp["elo"]))
    return elo_map, comp

def attach_elo_to_schedule(schedule_df: pd.DataFrame, elo_map: dict, team_abbr: str = TEAM) -> pd.DataFrame:
    """Normalize to abbreviations and attach elo_for/elo_against."""
    sch = schedule_df.copy()
    sch["home_team"] = sch["home_team"].map(NAME_TO_ABBR).fillna(sch["home_team"])
    sch["away_team"] = sch["away_team"].map(NAME_TO_ABBR).fillna(sch["away_team"])
    sch["home"] = (sch["home_team"] == team_abbr).astype(int)
    sch["opponent"] = np.where(sch["home"] == 1, sch["away_team"], sch["home_team"])
    sch["elo_for"] = np.where(sch["home"] == 1, sch["home_team"].map(elo_map), sch["away_team"].map(elo_map))
    sch["elo_against"] = np.where(sch["home"] == 1, sch["away_team"].map(elo_map), sch["home_team"].map(elo_map))
    cols = ["date","home_team","away_team","opponent","home","back_to_back","rest_days","rest_diff","elo_for","elo_against"]
    extra = [c for c in ["Location","Result","Match Number","Round Number"] if c in sch.columns]
    return sch[cols + extra]

def predict_schedule(sch_df: pd.DataFrame, backtest_csv: str,
                     n_sims: int = 0, rng_seed: int = 42,
                     use_elo_noise: bool = False, elo_noise_sd: float = 35.0):
    """
    Unified predictor.
    n_sims == 0 -> deterministic predictions with OTL allocation (closest-to-0.5).
    n_sims  > 0 -> simulations; display-case per game via MEAN wins + OTL allocated
                   among display losses to match backtest OTL share.
    Returns:
      if n_sims == 0: (preds_df, None, summary)
      else:           (display_df, sims_totals_df, summary)
    """
    sch = sch_df.copy().reset_index(drop=True)

    # ---------- base Elo prob (deterministic) ----------
    diff_base = (
        (sch["elo_for"].astype(float) - sch["elo_against"].astype(float))
        + HOME_EDGE * sch["home"].astype(int)
        - B2B_PENALTY * sch["back_to_back"].astype(int)
        + REST_PTS_PER_DAY * sch["rest_diff"].astype(float)
    ).values
    p_elo = 1.0 / (1.0 + 10.0 ** (-(diff_base) / ELO_SCALE_DEN))
    p_base = ALPHA_ELO * p_elo + (1.0 - ALPHA_ELO) * 0.5
    p_base = np.clip(p_base, 1e-6, 1 - 1e-6)

    # ---------- robust OTL share from backtest (SO counts as OTL) ----------
    bt = pd.read_csv(backtest_csv)

    # normalize result
    res = bt["result"].astype(str).str.strip().str.upper()

    # normalize extra_time and treat any non-"no" as beyond regulation
    ext = bt.get("extra_time", "no")
    ext = pd.Series(ext).fillna("no").astype(str).str.strip().str.lower()
    ext = ext.replace({
        "overtime": "ot", "otl": "ot", "ot/so": "ot",
        "shootout": "so", "shoot-out": "so",
        "": "no"
    })
    beyond = (ext != "no")
    if "so" in bt.columns:
        so_col = pd.to_numeric(bt["so"], errors="coerce").fillna(0).astype(int)
        beyond = beyond | (so_col == 1)

    wins_true = (res == "W").astype(int).values
    otl_true  = ((res == "L") & beyond).astype(int).values
    losses_true = int((wins_true == 0).sum())
    p_otl = (int(otl_true.sum()) / losses_true) if losses_true > 0 else 0.0
    p_otl = float(np.clip(p_otl, 0.0, 1.0))

    base_cols = [c for c in ["date","home_team","away_team","opponent","home",
                             "back_to_back","rest_days","rest_diff","elo_for","elo_against"]
                 if c in sch.columns]

    # ------------------ deterministic path ------------------
    if n_sims == 0:
        preds = sch[base_cols].copy()
        preds["win_prob"] = np.round(p_base, 3)
        pred_win = (p_base >= 0.5).astype(int)
        loss_idx = np.where(pred_win == 0)[0]

        # allocate OTLs among predicted losses by backtest share
        otl_idx = allocate_otl(loss_idx, p_base, p_otl) if len(loss_idx) else set()

        res_cat = np.array(["W"] * len(preds), dtype=object)
        if len(loss_idx):
            res_cat[loss_idx] = "L"
            if otl_idx:
                res_cat[list(otl_idx)] = "OTL"
        preds["predicted_result"] = res_cat

        pred_w  = int((preds["predicted_result"] == "W").sum())
        pred_ol = int((preds["predicted_result"] == "OTL").sum())
        pred_rl = int((preds["predicted_result"] == "L").sum())
        summary = {
            "n_games": len(preds),
            "pred_record_W-L-OTL": f"{pred_w}-{pred_rl}-{pred_ol}",
            "pred_points": int(2 * pred_w + pred_ol),
            "avg_win_prob": float(preds["win_prob"].mean()),
            "assumed_otl_share_from_backtest": round(p_otl, 3),
            "mode": "deterministic",
        }
        return preds, None, summary

    # ------------------ simulation path ------------------
    rng = np.random.default_rng(rng_seed)
    n_games = len(sch)
    wins_sims = np.zeros((n_sims, n_games), dtype=int)
    otl_sims  = np.zeros((n_sims, n_games), dtype=int)

    for s in range(n_sims):
        if use_elo_noise:
            diff = diff_base + rng.normal(0.0, elo_noise_sd, size=n_games)
            p_e = 1.0 / (1.0 + 10.0 ** (-(diff) / ELO_SCALE_DEN))
            p   = ALPHA_ELO * p_e + (1.0 - ALPHA_ELO) * 0.5
            p   = np.clip(p, 1e-6, 1 - 1e-6)
        else:
            p = p_base

        w = rng.binomial(1, p, size=n_games)
        wins_sims[s] = w

        loss_positions = np.where(w == 0)[0]
        if len(loss_positions) > 0 and p_otl > 0.0:
            otl_flags = rng.binomial(1, p_otl, size=len(loss_positions))
            otl_sims[s, loss_positions] = otl_flags

    # mean-rule for wins
    win_rate = wins_sims.mean(axis=0)

    # display-case: W by mean-rule; OTL allocated among remaining losses by backtest share
    display_result = np.full(n_games, "L", dtype=object)
    display_result[win_rate >= 0.5] = "W"

    remaining_losses = np.where(win_rate < 0.5)[0]
    otl_pick = allocate_otl(remaining_losses, p_base, p_otl) if len(remaining_losses) else set()
    if otl_pick:
        display_result[list(otl_pick)] = "OTL"

    display_df = sch[base_cols].copy()
    display_df["win_prob"] = np.round(p_base, 3)
    display_df["win_rate"] = np.round(win_rate, 3)
    display_df["display_result"] = display_result

    # per-sim totals (for histogram)
    rows = []
    for s in range(n_sims):
        w = int(wins_sims[s].sum())
        losses_idx = np.where(wins_sims[s] == 0)[0]
        ol = int(otl_sims[s, losses_idx].sum()) if len(losses_idx) else 0
        rl = int(len(losses_idx) - ol)
        pts = int(2 * w + ol)
        rows.append({"sim": s, "wins": w, "reg_losses": rl, "otl": ol, "points": pts,
                     "record_W-L-OTL": f"{w}-{rl}-{ol}"})
    sims_totals = pd.DataFrame(rows)

    # summary
    disp_w  = int((display_df["display_result"] == "W").sum())
    disp_ol = int((display_df["display_result"] == "OTL").sum())
    disp_rl = int((display_df["display_result"] == "L").sum())
    summary = {
        "n_games": n_games,
        "display_record_W-L-OTL": f"{disp_w}-{disp_rl}-{disp_ol}",
        "display_points": int(2 * disp_w + disp_ol),
        "mean_points": float(sims_totals["points"].mean()),
        "median_points": float(sims_totals["points"].median()),
        "assumed_otl_share_from_backtest": round(p_otl, 3),
        "n_sims": n_sims,
        "mode": "simulation_mean_rule",
    }
    return display_df, sims_totals, summary


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Leafs Season Predictor", layout="wide")
if LOGO_PATH.exists():
    st.image(str(LOGO_PATH), width=120)
else:
    st.text("nope")
st.title("Toronto Maple Leafs — 2025/2026 Season Prediction Dashboard")
# Default paths (relative to repo root). Adjust in sidebar as needed.
BASE = Path(__file__).resolve().parent.parent  # project root = one level up from /dashboard
default_schedule = str(BASE / "data" / "raw" / "schedule.csv")
default_teams    = str(BASE / "data" / "raw" / "teams.csv")
default_savepct  = str(BASE / "data" / "clean" / "team_save_percentages.csv")
default_backtest = str(BASE / "data" / "raw" / "backtest.csv")

with st.sidebar:
    st.header("Data paths")
    schedule_csv = st.text_input("Schedule CSV", value=default_schedule)
    teams_csv    = st.text_input("Teams CSV", value=default_teams)
    savepct_csv  = st.text_input("Team Save% CSV", value=default_savepct)
    backtest_csv = st.text_input("Backtest CSV", value=default_backtest)

    st.header("Simulation")
    n_sims = st.number_input("Number of simulations", min_value=0, max_value=5000, value=100, step=50)
    use_noise = st.checkbox("Add Elo noise per sim?", value=False)
    noise_sd  = st.number_input("Elo noise SD", min_value=0.0, value=35.0, step=5.0)

    st.header("Elo Adjustments:")
    home_edge = st.number_input("Home Bonus:", value=float(HOME_EDGE), step=1.0)
    b2b_pen   = st.number_input("Back-to-back Penalty", value=float(B2B_PENALTY), step=1.0)
    rest_pts  = st.number_input("Rest Day Bonus (Per Day)", value=float(REST_PTS_PER_DAY), step=0.5)
    elo_den   = 600.00
    alpha_elo = st.slider("ALPHA_ELO (weight on Elo)", min_value=0.0, max_value=1.0, value=float(ALPHA_ELO), step=0.05)

# update global knobs from UI (simple approach)
HOME_EDGE = home_edge
B2B_PENALTY = b2b_pen
REST_PTS_PER_DAY = rest_pts
ELO_SCALE_DEN = elo_den
ALPHA_ELO = alpha_elo

# Run
run = st.button("Run Predictions", type="primary")
if run:
    try:
        schedule_df = load_leafs_schedule_from_csv(schedule_csv)
        elo_map, teams_elos_df = build_elo_map(teams_csv, savepct_csv)
        sch_with_elos = attach_elo_to_schedule(schedule_df, elo_map, TEAM)

        results_df, sims_df, summary = predict_schedule(
            sch_df=sch_with_elos,
            backtest_csv=backtest_csv,
            n_sims=int(n_sims),
            rng_seed=7,
            use_elo_noise=bool(use_noise),
            elo_noise_sd=float(noise_sd),
        )

        c1, c2, c3 = st.columns([2, 1, 1])

        with c1:
            st.subheader("Per-game predictions")
            st.dataframe(results_df, use_container_width=True)

        with c2:
            st.subheader("Summary")
            if summary.get("mode") == "deterministic":
                st.metric("Predicted Record (W-L-OTL)", summary["pred_record_W-L-OTL"])
                st.metric("Predicted Points", summary["pred_points"])
                st.write(f"Avg win prob: {summary['avg_win_prob']:.3f}")
            else:
                st.metric("Projected Record (W-L-OTL)", summary["display_record_W-L-OTL"])
                st.metric("Projected Number of Points", summary["display_points"])

        with c3:
            if sims_df is not None and len(sims_df):
                st.subheader("Points distribution")

                # Points -> frequency
                counts = sims_df["points"].value_counts().sort_index()
                hist_df = counts.reset_index()
                hist_df.columns = ["points", "simulations"]

                chart = (
                    alt.Chart(hist_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("points:Q", title="Points"),
                        y=alt.Y("simulations:Q", title="Number of simulations"),
                        tooltip=["points", "simulations"]
                    )
                    .properties(height=300)
                )
                st.altair_chart(chart, use_container_width=True)


        # Downloads
        st.download_button(
            label="Download per-game predictions CSV",
            data=results_df.to_csv(index=False),
            file_name="predictions.csv",
            mime="text/csv",
        )
        if sims_df is not None and len(sims_df):
            st.download_button(
                label="Download simulations CSV",
                data=sims_df.to_csv(index=False),
                file_name="simulations.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error(f"Error: {e}")


#streamlit run dashboard/app.py
