import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading raw dataset...")
# REPLACE 'raw_kaggle_data.csv' with your actual filename!
df = pd.read_csv('ball_by_ball_ipl.csv') 

print("\n" + "="*50)
print("CHECKPOINT 1: Original Dataset Details")
print("="*50)
print(f"Total Columns ({len(df.columns)}): {list(df.columns)}")
print(f"Dataset Shape: {df.shape}")
print("="*50 + "\n")

print("Mapping your specific dataset columns...")
# 1. Your dataset uses 'Bat First' and 'Bat Second'. 
# We need to figure out who is batting based on the 'Innings' number.
df = df[df['Innings'].isin([1, 2])] # Keep only standard innings, ignore super overs
df['batting_team'] = np.where(df['Innings'] == 1, df['Bat First'], df['Bat Second'])
df['bowling_team'] = np.where(df['Innings'] == 1, df['Bat Second'], df['Bat First'])

# 2. Rename your specific columns to what our ML model expects
df.rename(columns={
    'Match ID': 'id',
    'Innings': 'inning',
    'Over': 'over',
    'Ball': 'ball',
    'Venue': 'city',
    'Runs From Ball': 'total_runs',
    'Player Out': 'player_dismissed'
}, inplace=True)

print("\n" + "="*50)
print("CHECKPOINT 2: After Renaming Columns")
print("="*50)
print(f"Mapped Columns: {list(df.columns)}")
print("="*50 + "\n")

print("Filtering for current active IPL teams...")
# 3. Filter for current active teams only
current_teams = [
    'Chennai Super Kings', 'Delhi Capitals', 'Gujarat Titans',
    'Kolkata Knight Riders', 'Lucknow Super Giants', 'Mumbai Indians',
    'Punjab Kings', 'Rajasthan Royals', 'Royal Challengers Bengaluru',
    'Sunrisers Hyderabad'
]

# Standardize older team names
df['batting_team'] = df['batting_team'].replace('Delhi Daredevils', 'Delhi Capitals')
df['bowling_team'] = df['bowling_team'].replace('Delhi Daredevils', 'Delhi Capitals')
df['batting_team'] = df['batting_team'].replace('Kings XI Punjab', 'Punjab Kings')
df['bowling_team'] = df['bowling_team'].replace('Kings XI Punjab', 'Punjab Kings')
df['batting_team'] = df['batting_team'].replace('Royal Challengers Bangalore', 'Royal Challengers Bengaluru')
df['bowling_team'] = df['bowling_team'].replace('Royal Challengers Bangalore', 'Royal Challengers Bengaluru')

# Keep only rows with current teams
df = df[df['batting_team'].isin(current_teams)]
df = df[df['bowling_team'].isin(current_teams)]

print("Calculating current scores, wickets, and overs...")
# Sort by match and innings to ensure correct rolling calculations
df = df.sort_values(['id', 'inning', 'over', 'ball'])

# Calculate cumulative runs per match per innings to get 'current_score'
df['current_score'] = df.groupby(['id', 'inning'])['total_runs'].cumsum()

# Calculate wickets fallen
# Fill NaN in player_dismissed with 0, and actual names with 1
df['player_dismissed'] = df['player_dismissed'].fillna("0")
df['player_dismissed'] = df['player_dismissed'].apply(lambda x: x if x == "0" else "1")
df['player_dismissed'] = df['player_dismissed'].astype('int')
df['wickets_fallen'] = df.groupby(['id', 'inning'])['player_dismissed'].cumsum()
df['wickets_left'] = 10 - df['wickets_fallen']

# Your dataset handily already has a 'Balls Remaining' column! Let's use it directly.
df['balls_left'] = df['Balls Remaining']
df['balls_bowled'] = 120 - df['balls_left']

print("Calculating Current Run Rate (CRR)...")
# Calculate CRR (Current Run Rate)
df['crr'] = (df['current_score'] * 6) / df['balls_bowled']
df['crr'] = df['crr'].replace([np.inf, -np.inf], 0).fillna(0) # Handle division by zero on the first ball

print("Calculating runs in the last 5 overs...")
# Feature Engineering: Runs in last 5 overs (rolling window of 30 balls)
groups = df.groupby(['id', 'inning'])
df['last_five'] = groups['total_runs'].rolling(window=30).sum().values
df.dropna(subset=['last_five'], inplace=True)

print("Determining total match scores...")
# Extracting the final target variable ('total')
total_score_df = df.groupby(['id', 'inning']).sum()['total_runs'].reset_index()
total_score_df.rename(columns={'total_runs': 'total'}, inplace=True)
df = df.merge(total_score_df, on=['id', 'inning'])

print("Finalizing dataset...")
# Drop missing city rows
df.dropna(subset=['city'], inplace=True)

# Final Cleanup: Select only the columns the ML model expects
final_df = df[['batting_team', 'bowling_team', 'city', 'current_score', 'balls_left', 'wickets_left', 'crr', 'last_five', 'total']]

# Shuffle the dataset to prevent the model from learning chronological biases
final_df = final_df.sample(frac=1)

# Save to CSV
final_df.to_csv('cleaned_ipl_data.csv', index=False)

print("\n" + "="*50)
print("CHECKPOINT 3: Final Processed Dataset")
print("="*50)
print(f"Final Features ({len(final_df.columns)}): {list(final_df.columns)}")
print(f"Final Dataset Shape: {final_df.shape}")
print("\nHere is a quick peek at the first 3 rows:")
print(final_df.head(3))
print("="*50 + "\n")

print("Success! 'cleaned_ipl_data.csv' has been generated and is ready for the ML model.")

print("\n" + "="*50)
print("CHECKPOINT 4: Generating Visuals...")
print("="*50)
try:
    plt.figure(figsize=(16, 6))
    sns.set_theme(style="whitegrid")
    
    # Graph 1: Target Variable Distribution
    plt.subplot(1, 2, 1)
    sns.histplot(final_df['total'], bins=35, kde=True, color='purple')
    plt.title('Distribution of Total Match Scores', fontsize=14, fontweight='bold')
    plt.xlabel('Total Score Evaluated', fontsize=12)
    plt.ylabel('Frequency (Number of matches)', fontsize=12)
    
    plt.savefig('checkpoints_visualized.png', dpi=300)
    print("Visuals generated successfully!")
except ImportError:
    print("Could not generate visualizations. Make sure to run: pip install matplotlib seaborn")
except Exception as e:
    print(f"An error occurred while plotting: {e}")