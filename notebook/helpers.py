# import libraries - updated
import pandas as pd 
import numpy as np
from datetime import datetime
import json 
import sys
import os 

POSITIONS_IN_MOTION = ['WR', 'TE', 'FB', 'RB']
MOTION_COLS = ['inMotionAtBallSnap', 'motionSinceLineset', 'shiftSinceLineset']
ADDITIONAL_NEW_COLS = ['_winnerTeamAbbr', '_playSuccessful', 'any_motion', '_is_pass', '_is_run', 'redZone']
SHOULD_NOT_BE_NULL_COLS = ['gameId', 'playId', 'nflId']
TIME_BUCKET = 180 # 3 min * 60 = 180 seconds

BASECDIR = "../dataset/kaggle/"
PREPARED_DF_FULL_FILE_PATH = "../dataset/saved/game-play-player.csv"


"""
    Calculate Red Zones (last 20 yards) in the field
"""
def calculate_red_zone(playdf):
    # Function to determine if the play is in the red zone
    def is_red_zone(row):
        # Check if the play is in the red zone based on the direction
        if row['playDirection'] == 'left' and row['absoluteYardlineNumber'] >= 10 and row['absoluteYardlineNumber'] <= 30:
            return True
        elif row['playDirection'] == 'right' and row['absoluteYardlineNumber'] >= 90 and row['absoluteYardlineNumber'] <= 110:
            return True
        else:
            return False

    # Apply the function to the dataframe
    playdf['redZone'] = playdf.apply(is_red_zone, axis=1)
    #return playdf

"""
    Normalize GameClock within each game

    GameClock resets at the start of each quarter, and the minimum valid value is 00:41, while the maximum is 14:56
    To normalize the gameClock, we need to:  
    - Add accomulative time from previous quarters.  
    Each quarter contribtues up to 15 minutes (900 seconds) of play time 
    - Normalize Across Quarters:  
    Compute total seconds relative to the start of the game (global game clock)

    Calculations 
    - Quarter contribution: (quarter - 1) * 900 (converts quarter to seconds; -1 since the first quarter adds no time)
    - GameClock in Seconds: minutes * 60 + seconds 
    - Adjusted Time: Quarter contribution + GameClock in Seconds

"""
def normalize_game_time(row):
    game_clock = row['gameClock']
    quarter = row['quarter']
    minutes, seconds = map(int, game_clock.split(':'))
    game_time = (quarter - 1) * 900 + (minutes * 60 + seconds)
    return game_time



"""
    Merge Tracking data with Play data (add time and PlayDirection columns)
"""
def merge_tracking_first_row_in_playdf(playdf):
    # List of tracking file paths
    tracking_paths = [f"../dataset/kaggle/tracking_week_{i}.csv" for i in range(1, 10)]

    # Initialize an empty list to store the first record DataFrames
    tracking_data = []

    # Process each tracking file
    for path in tracking_paths:
        # Read the tracking file
        tracking_df = pd.read_csv(path)
        
        # Extract the first matching record for each gameId and playId
        first_tracking_record = tracking_df.groupby(['gameId', 'playId']).first().reset_index()
        
        # Select only the necessary columns for the join
        columns_to_merge = ['gameId', 'playId', 'time', 'playDirection']
        tracking_data.append(first_tracking_record[columns_to_merge])

        # Concatenate all the processed tracking data
        merged_tracking_df = pd.concat(tracking_data, ignore_index=True)

        # Merge playdf with the combined tracking data
        result_df = playdf.merge(merged_tracking_df, on=['gameId', 'playId'], how='left')

        # Display the resulting dataframe
        result_df.shape


    return result_df



"""
Load data into unified dataframe and add required columns
Additional Columns:
    _winnerTeamAbbr
    _playSuccessful
    any_motion
Return:
    df
"""
def prepare_and_load_df(loadFromFile=False):
    if not loadFromFile: 
        print(f"Rebuild the File and Save it in {PREPARED_DF_FULL_FILE_PATH}")

        gamedf = pd.read_csv(f"{BASECDIR}games.csv")
        playdf = pd.read_csv(f"{BASECDIR}plays.csv")
        playerdf = pd.read_csv(f"{BASECDIR}players.csv")
        playerplaydf = pd.read_csv(f"{BASECDIR}player_play.csv")

        # Add winner team columns
        gamedf['_winnerTeamAbbr'] = np.where(gamedf['homeFinalScore'] > gamedf['visitorFinalScore'],
                                            gamedf['homeTeamAbbr'], 
                                            gamedf['visitorTeamAbbr'])
        # Add Successful play column to playdf 
        playdf['_playSuccessful'] = (
            ((playdf['yardsGained'] > 0) | (playdf['expectedPointsAdded'] > 0)) 
            # &
            # ((playdf['passResult'] == "C") | (playdf['hadRushAttempt'] == 1)) &
            # # (playdf['playNullifiedByPenalty'] == "N") &
            # # (playdf['sackYardsOffense'] == 0) &
            # (playdf['passResult'] != "IN")  
            #& (playdf['tackleForALossYardage'].fillna(0) == 0)
        ).astype(int)

        playdf.dropna(subset=['gameClock'], inplace=True)
        # Apply gameClock time normalization (quarter - 1) * 900 + (minutes * 60 + seconds)
        playdf['_global_game_time'] = playdf.apply(normalize_game_time, axis=1)
        # Group into time buckets (e.g., 3-minute intervals within each game)
        playdf['_game_time_bucket'] = playdf['_global_game_time'] // TIME_BUCKET  # 3-minute intervals within each game

        playdf['_is_pass']  = playdf['passResult'].notnull().astype(int)  # Add the new column playdf['_is_pass'] 
        playdf['_is_run']  = playdf['rushLocationType'].notnull().astype(int)  # Add the new column playdf['_is_pass'] 

        # Merge Playdf with Tracking records to get PlayDirection and Time 
        # ONLY add the first record for each gameId, playId to get the play direction
        result_df = merge_tracking_first_row_in_playdf(playdf)

        # Add redzone column to playdf 
        calculate_red_zone(result_df)

        print("Merge Data Frames")
        # Merge the dataframes if the file doesn't exist
        game_play_df = pd.merge(gamedf, result_df, on='gameId', how='inner')
        game_play_player_df = pd.merge(game_play_df, playerplaydf, on=['gameId', 'playId'], how='inner')
        df = pd.merge(game_play_player_df, playerdf, on='nflId', how='inner')

        # Drop rows where any value in the specified columns is null or NaN
        df.dropna(subset=SHOULD_NOT_BE_NULL_COLS, inplace=True)

        # Create a derived column indicating if any motion occurred
        df['any_motion'] = df[MOTION_COLS].apply(
            lambda row: row.any() if not row.isnull().all() else pd.NA, axis=1
        )

        # Merge with tracking data 
        # final_df = pd.merge(full_data_df, tracking_df, on=['gameId', 'playId', 'nflId'], how='inner')

        # Save the final merged dataset to CSV
        df.to_csv(PREPARED_DF_FULL_FILE_PATH, index=False)
                # # Check if the file exists
    else:
        print('Load from path!')
        # Load the CSV file into df if it exists
        df = pd.read_csv(PREPARED_DF_FULL_FILE_PATH)
    return df 




"""
Assuming here that the data is a merge between games df and plays df 
Select subset of dataframe based on gameId, offensive/deffensive, quarter, and winning team
"""
def selectOffenseDeffenseTeams(df, gameId=None, offensiveTeam=None, defensiveTeam=None, 
                               quarter=None, winningTeam=None):
    cond = df.playId > 0 

    # If gameId is provided, add the condition for gameId
    if gameId is not None:
        cond &= df['gameId'] == gameId

    # If offensiveTeam is provided, add the condition for offensiveTeam
    if offensiveTeam is not None:
        cond &= df['possessionTeam'] == offensiveTeam

    # If defensiveTeam is provided, add the condition for defensiveTeam
    if defensiveTeam is not None:
        cond &= df['defensiveTeam'] == defensiveTeam
    
    if quarter is not None:
        cond &= df['quarter'] == quarter

    if winningTeam is not None:
        if '_winnerTeamAbbr' in df.columns:
            cond &= df['_winnerTeamAbbr'] == winningTeam # offensiveTeam

    # Return the filtered DataFrame
    return df[cond].copy()


"""
Minimum Teams dataframe
"""
def print_game_info(df):
    # Group by gameId and retrieve the necessary columns
    game_info = df.groupby('gameId').agg(
        home_team=('homeTeamAbbr', 'first'),
        visitor_team=('visitorTeamAbbr', 'first'),
        home_score=('homeFinalScore', 'first'),
        visitor_score=('visitorFinalScore', 'first'),
        winner_team=('_winnerTeamAbbr', 'first')
    ).reset_index()

    # Print the game information for each unique gameId
    for _, row in game_info.iterrows():
        print(f"Game ID: {row['gameId']}, "
              f"Home Team: {row['home_team']} ({row['home_score']}), "
              f"Visitor Team: {row['visitor_team']} ({row['visitor_score']}), "
              f"Winner: {row['winner_team']}")

# Call the function to print the game information for all gameIds in filtered_df
#print_game_info(filtered_df)


"""
Filter game player play by only positions in motion 
"""
def filter_by_positions_in_motion(df):
    filtered_df = df[df['position'].isin(POSITIONS_IN_MOTION)].copy()
    return filtered_df


"""
    shared function used across other functions to aggragate values 
"""
def src_playsdf_offense_deffense_agg(df):
        # Group data by offensive formation, receiver alignment, defensive coverage, and defensive type (man/zone)
    grouped_df = df.groupby(['offenseFormation', 'receiverAlignment', 'pff_passCoverage', 'pff_manZone']).agg(
        total_plays=('playId', 'count'),
        avg_yards_gained=('yardsGained', lambda x: x.mean(skipna=True)),   
        success_count=('_playSuccessful', 'sum'),  # it was mean average success rate
        formation_count=('yardsGained', 'count'),  # Count the number of occurrences for each combination       
        pass_freq=('_is_pass', 'sum'), 
        run_freq=('_is_run', 'sum')
    ).reset_index()

    grouped_df['pass_rate'] = ((grouped_df['pass_freq']/ grouped_df['formation_count'])*100).round(2)
    grouped_df['run_rate'] =  ((grouped_df['run_freq']/ grouped_df['formation_count'])*100).round(2)
    grouped_df['formation_perent'] = ((grouped_df['formation_count'] / grouped_df.shape[0])*100).round(2)
    grouped_df['success_rate'] = ((grouped_df['success_count']/ grouped_df['total_plays'])*100).round(2)

    # Step 3: Sort the results to highlight defensive weaknesses (e.g., highest avg yards gained)
    results_sorted = grouped_df.sort_values(by=['avg_yards_gained', 'success_rate'], ascending=False)
    return results_sorted

"""
    shared function used across other functions to aggragate values 
    sort by either: 
        motion_percentage or 
        success_rate to show best offensive strategies 
"""
def playsdf_offense_deffense_agg(df, sort_by='success_rate', ascending=False):
        # Group data by offensive formation, receiver alignment, defensive coverage, and defensive type (man/zone)
    grouped_df = df.groupby(['offenseFormation', 'receiverAlignment', 'pff_passCoverage', 'pff_manZone']).agg(
        total_plays=('playId', 'count'),

        avg_yards_gained=('yardsGained', lambda x: x.mean(skipna=True)),   
        success_count=('_playSuccessful', 'sum'),  # it was mean average success rate
        
        pass_freq=('_is_pass', 'sum'), 
        run_freq=('_is_run', 'sum'),

        motion_plays=('any_motion', 'sum'),  # Total plays with any motion
        avg_yards_motion=('yardsGained', lambda x: x[df.loc[x.index, 'any_motion']== True].mean(skipna=True)),  # Avg yards gained with motion
        avg_yards_no_motion=('yardsGained', lambda x: x[df.loc[x.index, 'any_motion'] == False].mean(skipna=True)),  # Avg yards gained without motion


    ).reset_index()

    grouped_df['pass_rate'] = ((grouped_df['pass_freq']/ grouped_df['total_plays'])*100).round(2)
    grouped_df['run_rate'] =  ((grouped_df['run_freq']/ grouped_df['total_plays'])*100).round(2)
    grouped_df['formation_perent'] = ((grouped_df['total_plays'] / grouped_df.shape[0])*100).round(2)
    grouped_df['success_rate'] = ((grouped_df['success_count']/ grouped_df['total_plays'])*100).round(2)

    # Add motion percentage column
    grouped_df['motion_percentage'] = (grouped_df['motion_plays'] / grouped_df['total_plays']) * 100



    if sort_by == 'success_rate':
        # Step 3: Sort the results to highlight defensive weaknesses (e.g., highest avg yards gained)
        results_sorted = grouped_df.sort_values(by=['avg_yards_gained', 'success_rate'], ascending=ascending)
    else: 
        # Sort by motion percentage
        results_sorted = grouped_df.sort_values(by='motion_percentage', ascending=ascending) #, inplace=True)
    
    return results_sorted


"""
    Filter data by offensive/deffensive strategies
"""
def filter_by_offdeff_strategy(df, offenseFormation=None, receiverAlignment=None, pff_passCoverage=None, pff_manZone=None):
        # Filter the data based on offenseFormation, receiverAlignment, passCoverage, and manZone if they are provided
    if offenseFormation is not None:
        df = df[df['offenseFormation'] == offenseFormation.upper()]
    if receiverAlignment is not None:
        df = df[df['receiverAlignment'] == receiverAlignment]

    # Filter defensive strategy if provided
    if pff_passCoverage is not None:
        df = df[df['pff_passCoverage'] == pff_passCoverage]
    if pff_manZone is not None:
        df = df[df['pff_manZone'] == pff_manZone]

    # If no data is left after filtering, show a message and return
    # if df.empty:
    #     print(f"No data available for the selected filters.")
    #     return
    return df 
