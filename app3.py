import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Load KenPom data
@st.cache_data
def load_data():
    return pd.read_csv("summary25.csv")

df = load_data()

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f4f8;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 6px;
    }
    .stSlider>div>div {
        color: #1d4ed8;
    }
    </style>
""", unsafe_allow_html=True)

# Team name mapping for mismatches
team_name_map = {
    "Michigan State": "Michigan St.",
    "Ole Miss": "Mississippi"
}

# Sweet 16 matchups
sweet_16_matchups = [
    ("Alabama", "BYU"),
    ("Florida", "Maryland"),
    ("Duke", "Arizona"),
    ("Texas Tech", "Arkansas"),
    ("Auburn", "Michigan"),
    ("Michigan State", "Ole Miss"),
    ("Houston", "Purdue"),
    ("Tennessee", "Kentucky")
]

corrected_matchups = [
    (team_name_map.get(a, a), team_name_map.get(b, b))
    for a, b in sweet_16_matchups
]

# GUI: pick from Sweet 16
st.title("NCAA Sweet 16 Matchup Simulator")
all_sweet16_teams = sorted(set([team for pair in corrected_matchups for team in pair]))
team_a = st.selectbox("Team A", all_sweet16_teams)
team_b_options = [b for a, b in corrected_matchups if a == team_a] + [a for a, b in corrected_matchups if b == team_a]
team_b = st.selectbox("Team B (only Sweet 16 matchups)", sorted(set(team_b_options)))
single_game_runs = st.slider("Simulations", 100, 10000, 1000, 100)

# Pace adjustment
pace_adjust = st.slider("Adjust Pace (%)", 50, 150, 100, 5)
pace_multiplier = pace_adjust / 100

if st.button("Simulate This Game"):
    a_stats = df[df["TeamName"] == team_a].iloc[0]
    b_stats = df[df["TeamName"] == team_b].iloc[0]
    # Pace sensitivity sweep
    st.write("### Pace Sensitivity Curve")
    pace_range = np.arange(0.8, 1.21, 0.05)
    win_probs = []
    spreads = []
    for pace in pace_range:
        a_exp = (a_stats["AdjOE"] + b_stats["AdjDE"]) / 2 * pace
        b_exp = (b_stats["AdjOE"] + a_stats["AdjDE"]) / 2 * pace
        diff_sim = np.random.normal(a_exp - b_exp, 11, 1000)
        win_probs.append(np.mean(diff_sim > 0))
        spreads.append(np.mean(diff_sim))

    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_xlabel("Pace Multiplier")
    ax1.set_ylabel("Win Probability", color=color)
    ax1.plot(pace_range, win_probs, color=color, marker='o', label="Win Probability")
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel("Expected Spread", color=color)
    ax2.plot(pace_range, spreads, color=color, marker='x', linestyle='--', label="Expected Spread")
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    st.pyplot(fig)
    a_stats = df[df["TeamName"] == team_a].iloc[0]
    b_stats = df[df["TeamName"] == team_b].iloc[0]

    a_exp_pts = (a_stats["AdjOE"] + b_stats["AdjDE"]) / 2
    b_exp_pts = (b_stats["AdjOE"] + a_stats["AdjDE"]) / 2

    a_exp_pts *= pace_multiplier
    b_exp_pts *= pace_multiplier

    exp_spread = a_exp_pts - b_exp_pts
    diffs = np.random.normal(exp_spread, 11, single_game_runs)
    win_prob = np.mean(diffs > 0)
    moneyline = -round(100 * win_prob / (1 - win_prob)) if win_prob >= 0.5 else round(100 * (1 - win_prob) / win_prob)

    st.markdown("#### Simulation Results")
    st.write(f"**Adjusted Pace:** {pace_adjust}% of default")
    st.write(f"**Expected Spread:** {exp_spread:.2f} points")
    st.write(f"**{team_a} Win Probability:** {win_prob:.2%}")
    st.write(f"**Model Moneyline for {team_a}:** {moneyline:+}")

    fig, ax = plt.subplots()
    ax.hist(diffs, bins=40, alpha=0.7)
    ax.axvline(0, color='red', linestyle='--')
    ax.set_title(f"Simulated Point Differential: {team_a} vs {team_b}")
    ax.set_xlabel("Point Differential (A - B)")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

# --- Tournament Simulator ---
def simulate_game_once(team_a, team_b, pace=1.0):
    a = df[df["TeamName"] == team_a].iloc[0]
    b = df[df["TeamName"] == team_b].iloc[0]
    a_pts = ((a["AdjOE"] + b["AdjDE"]) / 2) * pace
    b_pts = ((b["AdjOE"] + a["AdjDE"]) / 2) * pace
    diff = np.random.normal(a_pts - b_pts, 11)
    return team_a if diff > 0 else team_b

def calculate_moneyline(win_prob):
    return -round(100 * win_prob / (1 - win_prob)) if win_prob >= 0.5 else round(100 * (1 - win_prob) / win_prob)

def simulate_tournament_full(n_sims, pace=1.0):
    win_counts = defaultdict(int)
    all_results = []

    for _ in range(n_sims):
        r16 = [simulate_game_once(a, b, pace) for a, b in corrected_matchups]
        r8 = [simulate_game_once(r16[i], r16[i+1], pace) for i in range(0, 8, 2)]
        r4 = [simulate_game_once(r8[i], r8[i+1], pace) for i in range(0, 4, 2)]
        champ = simulate_game_once(r4[0], r4[1], pace)
        win_counts[champ] += 1
        all_results.append({"Round of 16": r16, "Elite 8": r8, "Final Four": r4, "Champion": champ})

    results_df = pd.DataFrame({"Team": win_counts.keys(), "Championships": win_counts.values()})
    results_df["Win %"] = results_df["Championships"] / n_sims
    results_df["Moneyline"] = results_df["Win %"].apply(calculate_moneyline)
    results_df = results_df.sort_values("Win %", ascending=False).reset_index(drop=True)

    brackets = pd.DataFrame(all_results)
    return results_df, brackets

st.subheader("Simulate Full Sweet 16 Tournament")
n_sims = st.slider("Number of Simulations", 100, 5000, 1000, 100)
pace_pct = st.slider("Pace Adjustment (%) for Tournament", 50, 150, 100, 5)
pace_mult = pace_pct / 100

if st.button("Run Tournament Simulation"):
    results, brackets = simulate_tournament_full(n_sims, pace=pace_mult)

    st.write("### Championship Win Probabilities")
    st.dataframe(results)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(results["Team"], results["Win %"])
    ax.set_ylabel("Win Probability")
    ax.set_title("Simulated Tournament Championship Odds")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.download_button("Download Championship Probabilities (CSV)", data=results.to_csv(index=False), file_name="championship_odds.csv", mime="text/csv")
    st.download_button("Download Simulated Brackets (CSV)", data=brackets.to_csv(index=False), file_name="simulated_brackets.csv", mime="text/csv")

st.markdown("Upload `summary25.csv` to run with real data.")
