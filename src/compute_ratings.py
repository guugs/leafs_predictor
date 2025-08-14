import pandas as pd

def compute_team_rating(team_df, goalie_df):
    # Placeholder: combine metrics into composite rating
    # Example: weighted sum of 5v5 xGF%, PP%, PK%, and goalie xG saved above expected
    team_df['rating'] = (
        team_df['xGF%'] * 0.5 +
        team_df['PP_xG_rate'] * 0.2 +
        team_df['PK_xG_rate'] * 0.2 +
        goalie_df['xG_saved_above_expected'] * 0.1
    )
    return team_df[['team', 'rating']]
