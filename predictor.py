import numpy as np
import pandas as pd
import streamlit as st
import joblib
from pathlib import Path
from scipy.stats import poisson
from sklearn.preprocessing import LabelEncoder

DATA_FILE = "leagues_results_prev_2024-2025_and_current.xlsx"
MODEL_FILE = "models_bundle.pkl"

st.set_page_config(page_title="⚽ Predictor — Poisson + XGBoost", layout="wide")
st.title("⚽ Football Predictor — Poisson + XGBoost (pretrained)")

# ----------------------- Loaders -----------------------
@st.cache_data(show_spinner="Loading results…", ttl=3600)
def load_results(path: str) -> pd.DataFrame:
    """Load and stack all sheets that contain the required columns."""
    xls = pd.read_excel(path, sheet_name=None)
    frames = []
    needed = {"League","Season","Home Team","Away Team","Home Score","Away Score"}
    for name, df in xls.items():
        cols = set(map(str.strip, df.columns))
        if needed.issubset(cols) and len(df):
            frames.append(df[list(needed)])
    if not frames:
        return pd.DataFrame(columns=list(needed))
    out = pd.concat(frames, ignore_index=True)
    # clean types
    out["Home Score"] = pd.to_numeric(out["Home Score"], errors="coerce").fillna(0).astype(int)
    out["Away Score"] = pd.to_numeric(out["Away Score"], errors="coerce").fillna(0).astype(int)
    return out

@st.cache_resource(show_spinner="Loading models…")
def load_models_bundle(path: str):
    """
    models_bundle.pkl structure created earlier:
      { model_1x2, model_btts, model_hg, model_ag, label_classes, feature_names }
    """
    bundle = joblib.load(path)
    le = LabelEncoder()
    le.fit(bundle["label_classes"])
    models = (bundle["model_1x2"], bundle["model_btts"], bundle["model_hg"], bundle["model_ag"], le)
    feats  = bundle["feature_names"]
    return models, feats

# ----------------------- Helpers -----------------------
def team_strengths(df: pd.DataFrame):
    """Compute league-relative attack/defense for all teams in this df."""
    if df.empty:
        return {}
    lg_h = df["Home Score"].mean() or 1.0
    lg_a = df["Away Score"].mean() or 1.0
    stats = {}
    teams = set(df["Home Team"]).union(df["Away Team"])
    for t in teams:
        h = df[df["Home Team"]==t]
        a = df[df["Away Team"]==t]
        atk_h = (h["Home Score"].mean()/lg_h) if len(h) else 1.0
        def_h = (h["Away Score"].mean()/lg_a) if len(h) else 1.0
        atk_a = (a["Away Score"].mean()/lg_a) if len(a) else 1.0
        def_a = (a["Home Score"].mean()/lg_h) if len(a) else 1.0
        stats[t] = {
            "attack": float(np.nan_to_num((atk_h+atk_a)/2.0)),
            "defense": float(np.nan_to_num((def_h+def_a)/2.0))
        }
    return stats, float(lg_h), float(lg_a)

def make_feature_row(home, away, strengths, lg_h, lg_a, feature_names):
    """Build the feature vector required by the pretrained models."""
    ha = strengths.get(home, {}).get("attack", 0.0)
    hd = strengths.get(home, {}).get("defense", 0.0)
    aa = strengths.get(away, {}).get("attack", 0.0)
    ad = strengths.get(away, {}).get("defense", 0.0)
    base = {
        "Home_Attack": ha, "Home_Defense": hd,
        "Away_Attack": aa, "Away_Defense": ad,
        "LgAvgH": lg_h, "LgAvgA": lg_a
    }
    # order exactly as in training
    row = [base.get(f, 0.0) for f in feature_names]
    return pd.DataFrame([row], columns=feature_names, dtype=np.float32)

def poisson_matrix(hxg, axg, max_goals=6):
    M = np.zeros((max_goals+1, max_goals+1))
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            M[i,j] = poisson.pmf(i, hxg) * poisson.pmf(j, axg)
    return M

def market_probs(M):
    home = np.tril(M, -1).sum()
    draw = np.trace(M)
    away = np.triu(M, 1).sum()
    btts_yes = M[1:,1:].sum()
    over25 = M[np.add.outer(np.arange(M.shape[0]), np.arange(M.shape[1])) > 2].sum()
    return home, draw, away, btts_yes, over25

def poisson_verdict(h,d,a):
    if max(h,d,a) < 0.40: return "1X2"
    if d > 0.40: return "X"
    if h > 0.50 and d > 0.20: return "1X"
    if a > 0.50 and d > 0.20: return "X2"
    if h > a and h > 0.45: return "1"
    if a > h and a > 0.45: return "2"
    return "1X2"

def league_table(df):
    tbl = {}
    for _, r in df.iterrows():
        h,a,hs,as_ = r["Home Team"], r["Away Team"], int(r["Home Score"]), int(r["Away Score"])
        for t in [h,a]:
            if t not in tbl:
                tbl[t] = {"P":0,"W":0,"D":0,"L":0,"GF":0,"GA":0,"Pts":0}
        tbl[h]["P"]+=1; tbl[a]["P"]+=1
        tbl[h]["GF"]+=hs; tbl[h]["GA"]+=as_
        tbl[a]["GF"]+=as_; tbl[a]["GA"]+=hs
        if hs>as_: tbl[h]["W"]+=1; tbl[h]["Pts"]+=3; tbl[a]["L"]+=1
        elif hs<as_: tbl[a]["W"]+=1; tbl[a]["Pts"]+=3; tbl[h]["L"]+=1
        else: tbl[h]["D"]+=1; tbl[a]["D"]+=1; tbl[h]["Pts"]+=1; tbl[a]["Pts"]+=1
    if not tbl: 
        return pd.DataFrame(columns=["Team","P","W","D","L","GF","GA","Pts","GD"])
    df_tab = pd.DataFrame.from_dict(tbl, orient="index")
    df_tab["GD"] = df_tab["GF"] - df_tab["GA"]
    df_tab = df_tab.sort_values(by=["Pts","GD","GF"], ascending=False).reset_index().rename(columns={"index":"Team"})
    df_tab.index += 1
    return df_tab

# ----------------------- Load data & models -----------------------
data_path = Path(DATA_FILE)
model_path = Path(MODEL_FILE)

if not data_path.exists():
    st.error(f"Excel not found: {DATA_FILE}. Add it to the same folder as app.py.")
    st.stop()

if not model_path.exists():
    st.error(f"Model bundle not found: {MODEL_FILE}. Add it to the same folder as app.py.")
    st.stop()

df_all = load_results(data_path)
(models, feature_names) = load_models_bundle(model_path)
model_1x2, model_btts, model_hg, model_ag, label_encoder = models

if df_all.empty:
    st.error("Loaded Excel but found no valid rows.")
    st.stop()

# ----------------------- UI: choose league/season & teams -----------------------
pairs = sorted(df_all[["League","Season"]].drop_duplicates().itertuples(index=False), key=lambda x:(x[0],x[1]))
pretty_pairs = [f"{lg} — {sn}" for lg,sn in pairs]
st.subheader("Select dataset")
choice = st.selectbox("League & Season", pretty_pairs, index=0)

# filter df for selection
sel_league, sel_season = choice.split(" — ", 1)
df_sel = df_all[(df_all["League"]==sel_league) & (df_all["Season"]==sel_season)]

if df_sel.empty:
    st.warning("No matches in this selection yet.")
    st.stop()

strengths, lg_h, lg_a = team_strengths(df_sel)
teams_sorted = sorted(strengths.keys())
c1, c2 = st.columns(2)
home_team = c1.selectbox("Home Team", teams_sorted, index=0)
away_team = c2.selectbox("Away Team", [t for t in teams_sorted if t != home_team] or teams_sorted, index=0)

# ----------------------- Predict -----------------------
if home_team and away_team:
    # Poisson
    hxg = strengths[home_team]["attack"] * strengths[away_team]["defense"]
    axg = strengths[away_team]["attack"] * strengths[home_team]["defense"]
    M = poisson_matrix(hxg, axg)
    p1, px, p2, pbtts, pov25 = market_probs(M)
    i,j = np.unravel_index(np.argmax(M), M.shape)

    # XGBoost
    Xrow = make_feature_row(home_team, away_team, strengths, lg_h, lg_a, feature_names)
    p_1x2 = model_1x2.predict_proba(Xrow)[0]
    # label order mapping
    classes = list(label_encoder.classes_)
    prob_map = dict(zip(classes, p_1x2))
    x1 = prob_map.get("Home Win", 0.0)
    xx = prob_map.get("Draw", 0.0)
    x2 = prob_map.get("Away Win", 0.0)
    btts_yes = float(model_btts.predict_proba(Xrow)[0,1])
    phg = float(model_hg.predict(Xrow)[0]); pag = float(model_ag.predict(Xrow)[0])
    x_pred_score = f"{round(phg)} - {round(pag)}"
    x_ou = "Over 2.5" if (round(phg)+round(pag)) > 2.5 else "Under 2.5"
    x_btts_v = "Yes" if btts_yes > 0.5 else "No"

    tab1, tab2 = st.tabs(["📈 Match Predictor", "📊 League Table"])

    with tab1:
        st.subheader("🧠 Model Verdicts")
        a, b = st.columns(2)

        with a:
            st.markdown("**Poisson**")
            st.metric("1", f"{p1:.1%}")
            st.metric("X", f"{px:.1%}")
            st.metric("2", f"{p2:.1%}")
            st.metric("🎯 Most Likely Score", f"{i} - {j}")
            st.info(f"BTTS Yes: **{pbtts:.1%}**")
            st.info(f"Over 2.5: **{pov25:.1%}**")
            st.info(f"Suggested Bet: **{poisson_verdict(p1,px,p2)}**")

        with b:
            st.markdown("**XGBoost (pretrained)**")
            st.metric("1", f"{x1:.1%}")
            st.metric("X", f"{xx:.1%}")
            st.metric("2", f"{x2:.1%}")
            st.metric("🎯 Predicted Score", x_pred_score)
            st.info(f"BTTS Yes: **{btts_yes:.1%}**")
            st.info(f"Over/Under: **{x_ou}**  |  BTTS: **{x_btts_v}**")

    with tab2:
        st.dataframe(league_table(df_sel), use_container_width=True)

    st.caption(f"League averages for this selection → Home goals: {lg_h:.2f} | Away goals: {lg_a:.2f}")

