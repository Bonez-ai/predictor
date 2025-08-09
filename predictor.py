# app.py — Streamlit Cloud friendly (requests-based)
import re, time, datetime
from io import BytesIO
from urllib.parse import urljoin, urlparse
from collections import defaultdict

import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from scipy.stats import poisson

# ML
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────── UI & session ───────────────────────────
st.set_page_config(page_title="Football Predictor — BetExplorer + Poisson + XGBoost", layout="wide")
st.title("⚽ Football Predictor — BetExplorer + Poisson + XGBoost")

if 'predictions' not in st.session_state: st.session_state.predictions = []
if 'scraped_df'  not in st.session_state: st.session_state.scraped_df  = pd.DataFrame()
if 'trained_models' not in st.session_state: st.session_state.trained_models = None

LEAGUES = {
    "england":"England Premier League","england2":"England Championship","england3":"England League One","england4":"England League Two",
    "spain":"Spain La Liga","spain2":"Spain Segunda División",
    "germany":"Germany Bundesliga",
    "italy":"Italy Serie A","italy2":"Italy Serie B",
    "france":"France Ligue 1","france2":"France Ligue 2",
    "netherlands":"Netherlands Eredivisie","netherlands2":"Netherlands Eerste Divisie",
    "portugal":"Portugal Primeira Liga","portugal2":"Portugal Liga Portugal 2",
    "belgium":"Belgium Jupiler Pro League","belgium2":"Belgium Challenger Pro League",
    "switzerland":"Switzerland Super League","switzerland2":"Switzerland Challenge League",
    "austria":"Austria Bundesliga","austria2":"Austria 2. Liga",
    "denmark":"Denmark Superliga","denmark2":"Denmark 1st Division",
    "sweden":"Sweden Allsvenskan","sweden2":"Sweden Superettan",
    "norway":"Norway Eliteserien","norway2":"Norway 1. Division",
    "scotland":"Scotland Premiership","scotland2":"Scotland Championship",
    "turkey":"Turkey Süper Lig","turkey2":"Turkey 1. Lig",
    "spainB":"Spain Segunda División"
}

PREVIOUS_SEASONS = {
    "Belgium Jupiler Pro League 2024–2025": "https://www.betexplorer.com/football/belgium/jupiler-pro-league-2024-2025/results/",
    "Belgium Challenger Pro League 2024–2025": "https://www.betexplorer.com/football/belgium/challenger-pro-league-2024-2025/results/",
    "England Premier League 2024–2025": "https://www.betexplorer.com/football/england/premier-league-2024-2025/results/",
    "England Championship 2024–2025": "https://www.betexplorer.com/football/england/championship-2024-2025/results/",
    "England League One 2024–2025": "https://www.betexplorer.com/football/england/league-one-2024-2025/results/",
    "England League Two 2024–2025": "https://www.betexplorer.com/football/england/league-two-2024-2025/results/",
    "France Ligue 1 2024–2025": "https://www.betexplorer.com/football/france/ligue-1-2024-2025/results/",
    "France Ligue 2 2024–2025": "https://www.betexplorer.com/football/france/ligue-2-2024-2025/results/",
    "Germany Bundesliga 2024–2025": "https://www.betexplorer.com/football/germany/bundesliga-2024-2025/results/",
    "Germany 2. Bundesliga 2024–2025": "https://www.betexplorer.com/football/germany/2-bundesliga-2024-2025/results/",
    "Italy Serie A 2024–2025": "https://www.betexplorer.com/football/italy/serie-a-2024-2025/results/",
    "Italy Serie B 2024–2025": "https://www.betexplorer.com/football/italy/serie-b-2024-2025/results/",
    "Netherlands Eredivisie 2024–2025": "https://www.betexplorer.com/football/netherlands/eredivisie-2024-2025/results/",
    "Netherlands Eerste Divisie 2024–2025": "https://www.betexplorer.com/football/netherlands/eerste-divisie-2024-2025/results/",
    "Portugal Primeira Liga 2024–2025": "https://www.betexplorer.com/football/portugal/primeira-liga-2024-2025/results/",
    "Portugal Liga Portugal 2 2024–2025": "https://www.betexplorer.com/football/portugal/liga-portugal-2-2024-2025/results/",
    "Scotland Premiership 2024–2025": "https://www.betexplorer.com/football/scotland/premiership-2024-2025/results/",
    "Scotland Championship 2024–2025": "https://www.betexplorer.com/football/scotland/championship-2024-2025/results/",
    "Spain La Liga 2024–2025": "https://www.betexplorer.com/football/spain/laliga-2024-2025/results/",
    "Spain Segunda Division 2024–2025": "https://www.betexplorer.com/football/spain/segunda-division-2024-2025/results/",
    "Switzerland Super League 2024–2025": "https://www.betexplorer.com/football/switzerland/super-league-2024-2025/results/",
    "Switzerland Challenge League 2024–2025": "https://www.betexplorer.com/football/switzerland/challenge-league-2024-2025/results/",
    "Turkey Super Lig 2024–2025": "https://www.betexplorer.com/football/turkey/super-lig-2024-2025/results/",
    "Turkey 1. Lig 2024–2025": "https://www.betexplorer.com/football/turkey/1-lig-2024-2025/results/",
}

# ─────────────────────────── Requests helper ───────────────────────────
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123 Safari/537.36"

def get_soup(url: str) -> BeautifulSoup:
    """GET with retries + UA; return BeautifulSoup(lxml)."""
    sess = requests.Session()
    sess.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.8"})
    for attempt in range(4):
        r = sess.get(url, timeout=20)
        if r.status_code == 200 and r.text:
            return BeautifulSoup(r.text, "lxml")
        time.sleep(1.2 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url} (status={r.status_code})")

def _normalize_text(s: str) -> str:
    return (s or "").encode("ascii","ignore").decode().strip().lower()

def _country_slug_from_key(key: str) -> str:
    base = re.sub(r"\d+","", key.lower())
    special = {"southkorea":"south-korea","southafrica":"south-africa","hongkong":"hong-kong",
               "elsalvador":"el-salvador","saudiarabia":"saudi-arabia","usa":"usa"}
    return special.get(base, base)

def _ensure_results(url: str) -> str:
    if not url.endswith("/"): url += "/"
    if "results/" not in url: url = urljoin(url, "results/")
    return url

@st.cache_data(show_spinner="Finding league URL…", ttl=3600)
def find_league_results_url(country_key: str, pretty_name: str) -> str:
    slug = _country_slug_from_key(country_key)
    base = f"https://www.betexplorer.com/football/{slug}/"
    soup = get_soup(base)

    target = _normalize_text(pretty_name)
    best_href, best_score = None, -1
    for a in soup.select("a[href]"):
        t = _normalize_text(a.get_text(" ", strip=True))
        if not t: 
            continue
        # simple token overlap scoring
        score = sum(1 for tok in target.split() if tok and tok in t)
        if score > best_score:
            best_score, best_href = score, a.get("href")

    if not best_href:
        raise RuntimeError(f"Could not find league link for {pretty_name} on {base}")

    league_url = urljoin(base, best_href if best_href.endswith("/") else best_href + "/")
    return _ensure_results(league_url)

@st.cache_data(show_spinner="Scraping match data…", ttl=3600)
def scrape_results_page(url: str) -> pd.DataFrame:
    """Parse BetExplorer results page into a dataframe. (No Selenium)"""
    soup = get_soup(url)
    rows = []

    # BetExplorer lists results within <tr> rows; same structure as with Selenium version
    for tr in soup.select("tr"):
        a_match = tr.select_one("td.h-text-left a.in-match")
        a_score = tr.select_one("td.h-text-center a")
        if not a_match or not a_score:
            continue

        score_text = a_score.get_text(strip=True)
        if not re.fullmatch(r"\d+\s*:\s*\d+", score_text):
            continue

        teams_text = a_match.get_text(" ", strip=True)
        if " - " in teams_text:
            home, away = [t.strip() for t in teams_text.split(" - ", 1)]
        else:
            spans = a_match.select("span")
            if len(spans) >= 2:
                home, away = spans[0].get_text(strip=True), spans[-1].get_text(strip=True)
            else:
                continue

        hg, ag = [int(x) for x in score_text.replace(" ", "").split(":")]
        res = "Home Win" if hg > ag else "Away Win" if ag > hg else "Draw"

        rows.append({
            "Home Team": home,
            "Away Team": away,
            "Home Score": hg,
            "Away Score": ag,
            "Result": res
        })

    return pd.DataFrame(rows)

# ─────────────────────────── Features & Models ───────────────────────────
def calculate_team_strengths(df: pd.DataFrame):
    stats = {}
    if df.empty: return stats
    lg_h = df['Home Score'].mean() or 1.0
    lg_a = df['Away Score'].mean() or 1.0
    for t in set(df['Home Team']).union(df['Away Team']):
        h = df[df['Home Team']==t]; a = df[df['Away Team']==t]
        atk_h = (h['Home Score'].mean()/lg_h) if len(h) else 1.0
        def_h = (h['Away Score'].mean()/lg_a) if len(h) else 1.0
        atk_a = (a['Away Score'].mean()/lg_a) if len(a) else 1.0
        def_a = (a['Home Score'].mean()/lg_h) if len(a) else 1.0
        stats[t] = {'attack': float(np.nan_to_num((atk_h+atk_a)/2.0)),
                    'defense': float(np.nan_to_num((def_h+def_a)/2.0))}
    return stats

def create_features(df: pd.DataFrame):
    df = df.copy()
    df['Total Goals'] = df['Home Score'] + df['Away Score']
    df['BTTS'] = ((df['Home Score']>0) & (df['Away Score']>0)).astype(int)
    stats = calculate_team_strengths(df)
    df['Home_Attack']  = df['Home Team'].apply(lambda x: stats.get(x,{}).get('attack',0.0))
    df['Home_Defense'] = df['Home Team'].apply(lambda x: stats.get(x,{}).get('defense',0.0))
    df['Away_Attack']  = df['Away Team'].apply(lambda x: stats.get(x,{}).get('attack',0.0))
    df['Away_Defense'] = df['Away Team'].apply(lambda x: stats.get(x,{}).get('defense',0.0))
    return df

@st.cache_resource(show_spinner="Training XGBoost models…")
def train_xgboost_models(df: pd.DataFrame):
    dfF = create_features(df)
    feats = ['Home_Attack','Home_Defense','Away_Attack','Away_Defense']
    X = dfF[feats].dropna()
    if len(X) < 30:
        raise ValueError("Not enough matches to train reliably (need ~30+).")

    # 1X2
    le = LabelEncoder()
    y_1x2 = le.fit_transform(dfF.loc[X.index,'Result'])
    model_1x2 = xgb.XGBClassifier(
        objective='multi:softprob', num_class=3, eval_metric='mlogloss',
        n_estimators=300, max_depth=3, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.9
    ).fit(X, y_1x2)

    # BTTS
    y_btts = dfF.loc[X.index,'BTTS']
    model_btts = xgb.XGBClassifier(
        objective='binary:logistic', eval_metric='logloss',
        n_estimators=300, max_depth=3, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.9
    ).fit(X, y_btts)

    # Goals
    y_hg = dfF.loc[X.index,'Home Score']
    model_hg = xgb.XGBRegressor(
        objective='reg:squarederror', n_estimators=400, max_depth=3,
        learning_rate=0.05, subsample=0.9, colsample_bytree=0.9
    ).fit(X, y_hg)

    y_ag = dfF.loc[X.index,'Away Score']
    model_ag = xgb.XGBRegressor(
        objective='reg:squarederror', n_estimators=400, max_depth=3,
        learning_rate=0.05, subsample=0.9, colsample_bytree=0.9
    ).fit(X, y_ag)

    return (model_1x2, model_btts, model_hg, model_ag, le)

def predict_with_xgboost(home_team, away_team, df, models):
    m1x2, mbtts, mhg, mag, le = models
    stats = calculate_team_strengths(df)
    X = pd.DataFrame([{
        'Home_Attack': stats.get(home_team,{}).get('attack',0.0),
        'Home_Defense': stats.get(home_team,{}).get('defense',0.0),
        'Away_Attack': stats.get(away_team,{}).get('attack',0.0),
        'Away_Defense': stats.get(away_team,{}).get('defense',0.0),
    }])
    p1x2 = m1x2.predict_proba(X)[0]
    label_map = dict(zip(range(3), le.inverse_transform([0,1,2])))
    probs = {label_map[i]: p1x2[i] for i in range(3)}
    return {
        'home_p': probs.get('Home Win',0.0),
        'draw_p': probs.get('Draw',0.0),
        'away_p': probs.get('Away Win',0.0),
        'btts_yes': float(mbtts.predict_proba(X)[0,1]),
        'pred_hg': float(mhg.predict(X)[0]),
        'pred_ag': float(mag.predict(X)[0]),
    }

# Poisson helpers
def poisson_matrix(hxg, axg, max_goals=6):
    M = np.zeros((max_goals+1, max_goals+1))
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            M[i,j] = poisson.pmf(i,hxg) * poisson.pmf(j,axg)
    return M

def market_probs(M):
    home = np.tril(M,-1).sum(); draw = np.trace(M); away = np.triu(M,1).sum()
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
    for _,r in df.iterrows():
        h,a,hs,as_ = r['Home Team'], r['Away Team'], int(r['Home Score']), int(r['Away Score'])
        for t in [h,a]: tbl.setdefault(t, {"P":0,"W":0,"D":0,"L":0,"GF":0,"GA":0,"Pts":0})
        tbl[h]["P"]+=1; tbl[a]["P"]+=1
        tbl[h]["GF"]+=hs; tbl[h]["GA"]+=as_; tbl[a]["GF"]+=as_; tbl[a]["GA"]+=hs
        if hs>as_: tbl[h]["W"]+=1; tbl[h]["Pts"]+=3; tbl[a]["L"]+=1
        elif hs<as_: tbl[a]["W"]+=1; tbl[a]["Pts"]+=3; tbl[h]["L"]+=1
        else: tbl[h]["D"]+=1; tbl[a]["D"]+=1; tbl[h]["Pts"]+=1; tbl[a]["Pts"]+=1
    df_tab = pd.DataFrame.from_dict(tbl, orient='index')
    if df_tab.empty: return pd.DataFrame(columns=["Team","P","W","D","L","GF","GA","Pts","GD"])
    df_tab["GD"] = df_tab["GF"] - df_tab["GA"]
    df_tab = df_tab.sort_values(by=["Pts","GD","GF"], ascending=False).reset_index()
    df_tab.index += 1
    df_tab.rename(columns={"index":"Team"}, inplace=True)
    return df_tab

# ─────────────────────────── UI — Step 1 ───────────────────────────
st.header("1) Scrape Data from BetExplorer")
mode = st.radio("Season Type", ["Current Season","Previous Season"], horizontal=True)

if mode=="Current Season":
    league_name = st.selectbox("Select Current League", list(LEAGUES.values()))
    code = [k for k,v in LEAGUES.items() if v==league_name][0]
    if st.button("🔎 Scrape Current Season Data"):
        st.session_state.scraped_df = pd.DataFrame(); st.session_state.trained_models=None
        try:
            url = find_league_results_url(code, league_name)
            df  = scrape_results_page(url)
            if df.empty: st.error("No data found.")
            else:
                st.session_state.scraped_df = df
                st.success(f"Scraped {len(df)} matches for {league_name}.")
        except Exception as e:
            st.error(f"Scrape error: {e}")
else:
    prev = st.selectbox("Select Previous Season", list(PREVIOUS_SEASONS.keys()))
    if st.button("🕰️ Scrape Previous Season Data"):
        st.session_state.scraped_df = pd.DataFrame(); st.session_state.trained_models=None
        try:
            df = scrape_results_page(_ensure_results(PREVIOUS_SEASONS[prev]))
            if df.empty: st.error("No data found.")
            else:
                st.session_state.scraped_df = df
                st.success(f"Scraped {len(df)} matches for {prev}.")
        except Exception as e:
            st.error(f"Scrape error: {e}")

st.divider()

# ─────────────────────────── UI — Step 2 ───────────────────────────
st.header("2) Train Prediction Models")
if not st.session_state.scraped_df.empty:
    st.info(f"Dataset size: {len(st.session_state.scraped_df)} matches.")
    if st.button("🧠 Train XGBoost Models"):
        try:
            st.session_state.trained_models = train_xgboost_models(st.session_state.scraped_df)
            st.success("Models trained successfully.")
        except Exception as e:
            st.error(f"Training error: {e}")
else:
    st.info("Scrape data first.")

st.divider()

# ─────────────────────────── UI — Step 3 ───────────────────────────
st.header("3) Make Predictions")
if st.session_state.trained_models is not None:
    df = st.session_state.scraped_df
    strengths = calculate_team_strengths(df)
    if not strengths: st.warning("Not enough data to compute team strengths.")
    else:
        c1,c2 = st.columns(2)
        with c1: home = st.selectbox("Home Team", sorted(strengths.keys()))
        with c2:
            other = sorted([t for t in strengths.keys() if t!=home]) or sorted(strengths.keys())
            away  = st.selectbox("Away Team", other)

        if home and away:
            # Poisson
            hxg = strengths[home]['attack']*strengths[away]['defense']
            axg = strengths[away]['attack']*strengths[home]['defense']
            M = poisson_matrix(hxg, axg)
            p1,px,p2,pbtts,pov25 = market_probs(M)
            i,j = np.unravel_index(np.argmax(M), M.shape)

            # XGB
            xg = predict_with_xgboost(home, away, df, st.session_state.trained_models)
            xg_total = round(xg['pred_hg']) + round(xg['pred_ag'])
            xg_ou = "Over 2.5" if xg_total > 2.5 else "Under 2.5"
            xg_btts = "Yes" if xg['btts_yes'] > 0.5 else "No"

            table = league_table(df)
            tab1,tab2,tab3 = st.tabs(["📈 Match Predictor","📊 League Table","📝 Saved Predictions"])

            with tab1:
                st.subheader("🧠 Model Verdicts")
                a,b = st.columns(2)
                with a:
                    st.markdown("**Poisson**")
                    st.metric("1", f"{p1:.1%}"); st.metric("X", f"{px:.1%}"); st.metric("2", f"{p2:.1%}")
                    st.metric("🎯 Most Likely Score", f"{i} - {j}")
                    st.info(f"BTTS Yes: **{pbtts:.1%}**")
                    st.info(f"Over 2.5: **{pov25:.1%}**")
                    st.info(f"Suggested Bet: **{poisson_verdict(p1,px,p2)}**")
                with b:
                    st.markdown("**XGBoost**")
                    st.metric("1", f"{xg['home_p']:.1%}"); st.metric("X", f"{xg['draw_p']:.1%}"); st.metric("2", f"{xg['away_p']:.1%}")
                    st.metric("🎯 Predicted Score", f"{round(xg['pred_hg'])} - {round(xg['pred_ag'])}")
                    st.info(f"BTTS Yes: **{xg['btts_yes']:.1%}**")
                    st.info(f"Over/Under: **{xg_ou}**"); st.info(f"BTTS Verdict: **{xg_btts}**")

                if st.button("💾 Save This Prediction"):
                    st.session_state.predictions.append({
                        'Date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                        'Home Team': home, 'Away Team': away,
                        'Poisson 1%': f"{p1:.1%}", 'Poisson X%': f"{px:.1%}", 'Poisson 2%': f"{p2:.1%}",
                        'Poisson Most Likely Score': f"{i} - {j}",
                        'Poisson Suggested Bet': poisson_verdict(p1,px,p2),
                        'Poisson BTTS Yes%': f"{pbtts:.1%}", 'Poisson Over2.5%': f"{pov25:.1%}",
                        'XGB 1%': f"{xg['home_p']:.1%}", 'XGB X%': f"{xg['draw_p']:.1%}", 'XGB 2%': f"{xg['away_p']:.1%}",
                        'XGB Pred Score': f"{round(xg['pred_hg'])} - {round(xg['pred_ag'])}",
                        'XGB BTTS Yes%': f"{xg['btts_yes']:.1%}", 'XGB OU Verdict': xg_ou
                    })
                    st.success("Prediction saved.")

            with tab2: st.dataframe(table, use_container_width=True)
            with tab3:
                if st.session_state.predictions:
                    dfp = pd.DataFrame(st.session_state.predictions)
                    st.dataframe(dfp, use_container_width=True)
                    st.download_button("📥 Download CSV", dfp.to_csv(index=False).encode("utf-8"),
                                       file_name="predictions.csv", mime="text/csv")
                    try:
                        out = BytesIO()
                        with pd.ExcelWriter(out, engine="openpyxl") as w: dfp.to_excel(w, index=False, sheet_name="Predictions")
                        st.download_button("📥 Download Excel", out.getvalue(),
                                           file_name="predictions.xlsx",
                                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                    except Exception:
                        st.caption("Install openpyxl for Excel export.")
                    if st.button("🗑️ Clear All Predictions"):
                        st.session_state.predictions = []; st.success("Cleared."); st.rerun()
                else:
                    st.info("No predictions saved yet.")
else:
    st.info("Complete steps 1 and 2 to enable predictions.")
