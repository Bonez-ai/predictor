import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from scipy.stats import poisson
from io import BytesIO
import datetime

# -----------------------------
# Initialize session state for predictions
# -----------------------------
if 'predictions' not in st.session_state:
    st.session_state.predictions = []

# -----------------------------
# League Configuration
# -----------------------------
LEAGUES = {
    "england": "England Premier League",
    "england2": "England Championship",
    "england3": "England League One",
    "england4": "England League Two",
    "england5": "England National League",
    "spain": "Spain La Liga",
    "spain2": "Spain Segunda División",
    "germany": "Germany Bundesliga",
    "italy": "Italy Serie A",
    "italy2": "Italy Serie B",
    "france": "France Ligue 1",
    "france2": "France Ligue 2",
    "france3": "France National",
    "netherlands": "Netherlands Eredivisie",
    "netherlands2": "Netherlands Eerste Divisie",
    "portugal": "Portugal Primeira Liga",
    "portugal2": "Portugal Segunda Liga",
    "belgium": "Belgium Pro League",
    "belgium2": "Belgium Challenger Pro League",
    "switzerland": "Switzerland Super League",
    "switzerland2": "Switzerland Challenge League",
    "austria": "Austria Bundesliga",
    "austria2": "Austria 2. Liga",
    "denmark": "Denmark Superliga",
    "denmark2": "Denmark 1st Division",
    "sweden": "Sweden Allsvenskan",
    "sweden2": "Sweden Superettan",
    "sweden3": "Sweden Ettan",
    "sweden4": "Sweden Division 2",
    "sweden11": "Sweden Women's League",
    "norway": "Norway Eliteserien",
    "norway2": "Norway 1. Division",
    "norway3": "Norway 2. Division",
    "finland": "Finland Veikkausliiga",
    "finland2": "Finland Ykkönen",
    "iceland": "Iceland Úrvalsdeild",
    "iceland2": "Iceland 1. deild",
    "scotland": "Scotland Premiership",
    "scotland2": "Scotland Championship",
    "ireland": "Ireland Premier Division",
    "ireland2": "Ireland First Division",
    "turkey": "Turkey Süper Lig",
    "turkey2": "Turkey 1. Lig",
    "greece": "Greece Super League",
    "greece2": "Greece Super League 2",
    "serbia": "Serbia SuperLiga",
    "brazil": "Brazil Série A",
    "brazil2": "Brazil Série B",
    "argentina": "Argentina Primera División",
    "uruguay": "Uruguay Primera División",
    "colombia": "Colombia Primera A",
    "colombia2": "Colombia Primera B",
    "chile": "Chile Primera División",
    "usa": "USA MLS",
    "usa2": "USA USL Championship",
    "japan": "Japan J1 League",
    "japan2": "Japan J2 League",
    "southkorea": "South Korea K League 1",
    "southkorea2": "South Korea K League 2",
    "china": "China Super League",
    "indonesia": "Indonesia Liga 1",
    "singapore": "Singapore Premier League",
    "hongkong": "Hong Kong Premier League",
    "southafrica": "South Africa Premier Division",
    "egypt": "Egypt Premier League",
    "saudiarabia": "Saudi Arabia Pro League",
    "israel": "Israel Premier League",
    "jamaica": "Jamaica Premier League",
    "elsalvador": "El Salvador Primera División",
    "albenia": "Albania Kategoria Superiore"
}

# -----------------------------
# Scraper
# -----------------------------
def scrape_soccer_results(league_code):
    url = f"https://www.soccerstats.com/results.asp?league={league_code}&pmtype=bydate"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.content, 'html.parser')
        table = soup.find('table', {'id': 'btable'})
        data = []
        for row in table.find_all('tr'):
            cols = row.find_all('td')
            if len(cols) >= 5:
                score = cols[2].get_text(strip=True)
                if ' - ' in score:
                    home_score, away_score = map(int, score.split(' - '))
                    data.append({
                        'Date': cols[0].get_text(strip=True),
                        'Home Team': cols[1].get_text(strip=True),
                        'Away Team': cols[3].get_text(strip=True),
                        'Home Score': home_score,
                        'Away Score': away_score,
                        'Result': 'Home Win' if home_score > away_score else 'Away Win' if away_score > away_score else 'Draw'
                    })
        return pd.DataFrame(data)
    except:
        return pd.DataFrame()

# -----------------------------
# Form Strength
# -----------------------------
def calculate_team_strengths(df):
    stats = {}
    for team in set(df['Home Team']).union(df['Away Team']):
        home = df[df['Home Team'] == team]
        away = df[df['Away Team'] == team]
        scored = home['Home Score'].sum() + away['Away Score'].sum()
        conceded = home['Away Score'].sum() + away['Home Score'].sum()
        games = len(home) + len(away)
        if games == 0: continue
        stats[team] = {'attack': scored / games, 'defense': conceded / games}
    return stats

# -----------------------------
# Poisson Matrix
# -----------------------------
def calculate_poisson_matrix(home_xg, away_xg, max_goals=6):
    matrix = np.zeros((max_goals+1, max_goals+1))
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            matrix[i, j] = poisson.pmf(i, home_xg) * poisson.pmf(j, away_xg)
    return pd.DataFrame(matrix, index=[f"Home {i}" for i in range(max_goals+1)], columns=[f"Away {j}" for j in range(max_goals+1)])

# -----------------------------
# Betting Strategy
# -----------------------------
def calculate_btts(matrix):
    yes = no = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if i > 0 and j > 0:
                yes += matrix.iat[i, j]
            else:
                no += matrix.iat[i, j]
    return {'BTTS Yes': yes, 'BTTS No': no}

def calculate_goals_markets(matrix):
    totals = {}
    for t in [1.5, 2.5, 3.5]:
        over = under = 0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                g = i + j
                p = matrix.iat[i, j]
                if g > t:
                    over += p
                else:
                    under += p
        totals[f'Over {t}'] = over
        totals[f'Under {t}'] = under
    return totals

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Poisson Predictor", layout="wide")
st.title("⚽ Poisson Football Predictor + Strategy")

league = st.selectbox("Select League", list(LEAGUES.values()))
code = [k for k, v in LEAGUES.items() if v == league][0]

with st.spinner("Fetching match data..."):
    df = scrape_soccer_results(code)

if df.empty:
    st.error("No results found. Try another league.")
    st.stop()

strengths = calculate_team_strengths(df)

col1, col2 = st.columns(2)
home = col1.selectbox("Home Team", sorted(strengths.keys()))
away = col2.selectbox("Away Team", sorted(strengths.keys()), index=1)

hxg = strengths[home]['attack'] * strengths[away]['defense']
axg = strengths[away]['attack'] * strengths[home]['defense']

matrix = calculate_poisson_matrix(hxg, axg)

st.subheader(f"📊 Score Probability Matrix: {home} vs {away}")
st.dataframe(matrix.style.format("{:.3f}"))

home_prob = np.tril(matrix.values, -1).sum()
draw_prob = np.trace(matrix.values)
away_prob = np.triu(matrix.values, 1).sum()
scoreline = np.unravel_index(np.argmax(matrix.values), matrix.shape)

st.subheader("🔮 Outcome Probabilities")
c1, c2, c3 = st.columns(3)
c1.metric("1", f"{home_prob:.1%}")
c2.metric("X", f"{draw_prob:.1%}")
c3.metric("2", f"{away_prob:.1%}")

st.metric("🎯 Most Likely Score", f"{scoreline[0]} - {scoreline[1]}")

# BTTS & Goals
st.subheader("🎯 Market Predictions")
btts = calculate_btts(matrix)
goals = calculate_goals_markets(matrix)

c1, c2 = st.columns(2)
with c1:
    st.metric("BTTS Yes", f"{btts['BTTS Yes']:.1%}")
    st.metric("BTTS No", f"{btts['BTTS No']:.1%}")
with c2:
    st.metric("Over 2.5", f"{goals['Over 2.5']:.1%}")
    st.metric("Under 2.5", f"{goals['Under 2.5']:.1%}")


# -----------------------------
# League Table Generator
# -----------------------------
def generate_league_table(results_df):
    table = {}
    for _, row in results_df.iterrows():
        home, away = row['Home Team'], row['Away Team']
        hs, as_ = int(row['Home Score']), int(row['Away Score'])

        for team in [home, away]:
            if team not in table:
                table[team] = {"P": 0, "W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0, "Pts": 0}

        table[home]["P"] += 1
        table[away]["P"] += 1
        table[home]["GF"] += hs
        table[home]["GA"] += as_
        table[away]["GF"] += as_
        table[away]["GA"] += hs

        if hs > as_:
            table[home]["W"] += 1
            table[home]["Pts"] += 3
            table[away]["L"] += 1
        elif hs < as_:
            table[away]["W"] += 1
            table[away]["Pts"] += 3
            table[home]["L"] += 1
        else:
            table[home]["D"] += 1
            table[away]["D"] += 1
            table[home]["Pts"] += 1
            table[away]["Pts"] += 1

    df_table = pd.DataFrame.from_dict(table, orient='index')
    df_table["GD"] = df_table["GF"] - df_table["GA"]
    df_table = df_table.sort_values(by=["Pts", "GD", "GF"], ascending=False).reset_index()
    df_table.index += 1
    df_table.rename(columns={"index": "Team"}, inplace=True)
    return df_table

# Generate table
league_table = generate_league_table(df)

# Team position info
def get_team_position(team, table):
    pos_row = table[table["Team"] == team]
    if not pos_row.empty:
        pos = pos_row.index[0]
        return f"{team} (#{pos})"
    return f"{team} (unranked)"

# Final prediction logic
def get_final_verdict(home_p, draw_p, away_p):
    if max(home_p, draw_p, away_p) < 0.4:
        return "1X2"
    elif draw_p > 0.4:
        return "X"
    elif home_p > 0.5 and draw_p > 0.2:
        return "1X"
    elif away_p > 0.5 and draw_p > 0.2:
        return "X2"
    elif home_p > 0.5 and away_p > 0.2:
        return "12"
    elif home_p > away_p and home_p > 0.45:
        return "1"
    elif away_p > home_p and away_p > 0.45:
        return "2"
    else:
        return "1X2"

# Add tabs
tab1, tab2, tab3 = st.tabs(["📈 Match Predictor", "📊 League Table", "📝 Saved Predictions"])


# -----------------------------
# Display Last 8-Match Goal Averages
# -----------------------------
def display_last_8_avg_goals(df, home_team, away_team):
    import streamlit as st

    def get_avg(team):
        recent = df[(df['Home Team'] == team) | (df['Away Team'] == team)].tail(8)
        total_goals = 0
        for _, match in recent.iterrows():
            if match['Home Team'] == team:
                total_goals += match['Home Score'] + match['Away Score']
            else:
                total_goals += match['Away Score'] + match['Home Score']
        avg = total_goals / len(recent) if len(recent) > 0 else 0
        return avg

    home_avg = get_avg(home_team)
    away_avg = get_avg(away_team)

    combined_avg = home_avg + away_avg
    verdict = "Over 2.5" if home_avg >= 3.0 and away_avg >= 3.0 and combined_avg >= 7.0 else "Under 2.5"

    st.subheader("📊 Last 8-Match Goal Averages")
    col1, col2 = st.columns(2)
    col1.metric(f"{home_team}", f"{home_avg:.2f} avg goals")
    col2.metric(f"{away_team}", f"{away_avg:.2f} avg goals")

    st.info(f"**Verdict: `{verdict}` based on 8-match averages**", icon="⚽")
    st.write(f"Combined average: {combined_avg:.2f}")

# -----------------------------
# Excel Export Function
# -----------------------------
def create_excel_download():
    if not st.session_state.predictions:
        return None
    
    df_predictions = pd.DataFrame(st.session_state.predictions)
    
    # Create Excel file in memory
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_predictions.to_excel(writer, index=False, sheet_name='Predictions')
    
    return output.getvalue()

with tab1:
    display_last_8_avg_goals(df, home, away)
    st.subheader("📍 Team Rankings")
    col1, col2 = st.columns(2)
    col1.success(get_team_position(home, league_table))
    col2.success(get_team_position(away, league_table))

    st.subheader("🧠 Model Verdict")
    verdict = get_final_verdict(home_prob, draw_prob, away_prob)
    st.info(f"**Suggested Bet: `{verdict}`**", icon="📌")
    
    # Save Prediction Button
    st.subheader("💾 Save Prediction")
    if st.button("Save This Prediction"):
        prediction_data = {
            'Date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            'League': league,
            'Home Team': home,
            'Away Team': away,
            'Home Win %': f"{home_prob:.1%}",
            'Draw %': f"{draw_prob:.1%}",
            'Away Win %': f"{away_prob:.1%}",
            'Most Likely Score': f"{scoreline[0]} - {scoreline[1]}",
            'Suggested Bet': verdict,
            'BTTS Yes %': f"{btts['BTTS Yes']:.1%}",
            'Over 2.5 %': f"{goals['Over 2.5']:.1%}",
            'Under 2.5 %': f"{goals['Under 2.5']:.1%}"
        }
        st.session_state.predictions.append(prediction_data)
        st.success("Prediction saved!")

with tab2:
    st.subheader(f"📊 {league} Table (Auto-Generated)")
    st.dataframe(league_table)

with tab3:
    st.subheader("📝 Saved Predictions")
    
    if st.session_state.predictions:
        # Display predictions
        df_display = pd.DataFrame(st.session_state.predictions)
        st.dataframe(df_display)
        
        # Download button
        excel_data = create_excel_download()
        if excel_data:
            st.download_button(
                label="📥 Download Predictions as Excel",
                data=excel_data,
                file_name=f"football_predictions_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        # Clear predictions button
        if st.button("🗑️ Clear All Predictions"):
            st.session_state.predictions = []
            st.success("All predictions cleared!")
            st.rerun()
    else:
        st.info("No predictions saved yet. Go to the Match Predictor tab to save predictions.")