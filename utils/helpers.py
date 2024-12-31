# import libraries - updated
import pandas as pd 
import numpy as np
from datetime import datetime
import json 
import sys
import os 

import boto3
from io import StringIO

S3_BUCKET_NAME = "nfl2025"
S3_FILE_KEY = "dataset/saved/game-play-player.csv"
BASECDIR = 'dataset/kaggle'
PREPARED_DF_FULL_FILE_PATH = "dataset/saved/game-play-player.csv"

POSITIONS_IN_MOTION = ['WR', 'TE', 'FB', 'RB']
MOTION_COLS = ['inMotionAtBallSnap', 'motionSinceLineset', 'shiftSinceLineset']
ADDITIONAL_NEW_COLS = ['_winnerTeamAbbr', '_playSuccessful', 'any_motion', '_is_pass', '_is_run', 'redZone']
SHOULD_NOT_BE_NULL_COLS = ['gameId', 'playId', 'nflId']
TIME_BUCKET = 180 # 3 min * 60 = 180 seconds




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
    # Initialize an empty list to store the first record DataFrames
    tracking_data = []

    # Process each tracking file from S3
    for i in range(1, 10):
        s3_file_key = f"{BASECDIR}/tracking_week_{i}.csv"
        
        # Read the tracking file from S3
        tracking_df = read_csv_from_s3(S3_BUCKET_NAME, s3_file_key)

        # Extract the first matching record for each gameId and playId
        first_tracking_record = tracking_df.groupby(['gameId', 'playId']).first().reset_index()

        # Select only the necessary columns for the join
        columns_to_merge = ['gameId', 'playId', 'time', 'playDirection']
        tracking_data.append(first_tracking_record[columns_to_merge])

    # Concatenate all the processed tracking data
    merged_tracking_df = pd.concat(tracking_data, ignore_index=True)

    # Merge playdf with the combined tracking data
    result_df = playdf.merge(merged_tracking_df, on=['gameId', 'playId'], how='left')

    return result_df


def read_csv_from_s3(bucket_name, file_key):
    """Read a CSV file from S3 and return a DataFrame."""

    # Extract the filename from the file_key
    filename = os.path.basename(file_key)
    
    # Define the local file path
    local_file_path = os.path.join('dataset', filename)
    
    # Check if the file exists locally
    if os.path.exists(local_file_path):
        return pd.read_csv(local_file_path)
    else:
        # Download the file from S3
        s3 = boto3.client('s3')
        obj = s3.get_object(Bucket=bucket_name, Key=file_key)
        data = obj['Body'].read().decode('utf-8')
        
        # Save the file locally
        with open(local_file_path, 'w') as file:
            file.write(data)
        
        # Read and return the DataFrame
        return pd.read_csv(StringIO(data))

def write_csv_to_s3(df, bucket_name, file_key):
    """Write a DataFrame to a CSV file in S3."""

    s3 = boto3.client('s3')
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    s3.put_object(Bucket=bucket_name, Key=file_key, Body=csv_buffer.getvalue())
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

        gamedf = read_csv_from_s3(S3_BUCKET_NAME, f"{BASECDIR}/games.csv")
        playdf = read_csv_from_s3(S3_BUCKET_NAME, f"{BASECDIR}/plays.csv")
        playerdf = read_csv_from_s3(S3_BUCKET_NAME, f"{BASECDIR}/players.csv")
        playerplaydf = read_csv_from_s3(S3_BUCKET_NAME, f"{BASECDIR}/player_play.csv")

        # Add winner team columns
        gamedf['_winnerTeamAbbr'] = np.where(gamedf['homeFinalScore'] > gamedf['visitorFinalScore'],
                                             gamedf['homeTeamAbbr'], 
                                             gamedf['visitorTeamAbbr'])
        # Add Successful play column to playdf 
        playdf['_playSuccessful'] = (
            ((playdf['yardsGained'] > 0) | (playdf['expectedPointsAdded'] > 0)) 
        ).astype(int)

        playdf.dropna(subset=['gameClock'], inplace=True)

        # Apply gameClock time normalization (quarter - 1) * 900 + (minutes * 60 + seconds)
        playdf['_global_game_time'] = playdf.apply(normalize_game_time, axis=1)

        # Group into time buckets (e.g., 3-minute intervals within each game)
        playdf['_game_time_bucket'] = playdf['_global_game_time'] // TIME_BUCKET  # 3-minute intervals within each game

        playdf['_is_pass'] = playdf['passResult'].notnull().astype(int)
        playdf['_is_run'] = playdf['rushLocationType'].notnull().astype(int)

        # Merge Playdf with Tracking records to get PlayDirection and Time 
        result_df = merge_tracking_first_row_in_playdf(playdf)

        # Add redzone column to playdf 
        calculate_red_zone(result_df)

        # Merge the dataframes
        game_play_df = pd.merge(gamedf, result_df, on='gameId', how='inner')
        game_play_player_df = pd.merge(game_play_df, playerplaydf, on=['gameId', 'playId'], how='inner')
        df = pd.merge(game_play_player_df, playerdf, on='nflId', how='inner')

        # Drop rows where any value in the specified columns is null or NaN
        df.dropna(subset=SHOULD_NOT_BE_NULL_COLS, inplace=True)

        # Create a derived column indicating if any motion occurred
        df['any_motion'] = df[MOTION_COLS].apply(
            lambda row: row.any() if not row.isnull().all() else pd.NA, axis=1
        )

        # Save the final merged dataset to S3
        write_csv_to_s3(df, S3_BUCKET_NAME, S3_FILE_KEY)
    else:
        # Load the CSV file from S3
        df = read_csv_from_s3(S3_BUCKET_NAME, S3_FILE_KEY)

    return df



"""
Assuming here that the data is a merge between games df and plays df 
Select subset of dataframe based on gameId, offensive/deffensive, quarter, and winning team
"""
def selectOffenseDeffenseTeams(df, 
                               gameId=None, 
                               offensiveTeam=None, 
                               defensiveTeam=None, 
                               quarter=None, 
                               winningTeam=None, 
                               offenseFormation=None, 
                               receiverAlignment=None, 
                               pff_passCoverage=None):
    
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
        cond &= df['quarter'] == int(quarter)

    if winningTeam is not None:
        if '_winnerTeamAbbr' in df.columns:
            cond &= df['_winnerTeamAbbr'] == winningTeam # offensiveTeam

    # Return the filtered DataFrame
    output_df = df[cond].copy()

    if offenseFormation is not None or receiverAlignment is not None or pff_passCoverage is not None: 
        output_df = filter_by_offdeff_strategy(output_df, offenseFormation, receiverAlignment, pff_passCoverage)

    return output_df 


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
    grouped_df = df.groupby(['offenseFormation', 'receiverAlignment', 'pff_passCoverage']).agg(
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
    
    # Create new columns for pass and run yards separately
    df['avg_yards_when_pass'] = df['yardsGained'] * df['_is_pass']
    df['avg_yards_when_run'] = df['yardsGained'] * df['_is_run']

    # Create new columns for yards when motion and no motion
    df['any_motion'] = df['any_motion'].astype(bool)
    df['avg_yards_when_motion'] = df['yardsGained'] * df['any_motion']
    df['avg_yards_when_no_motion'] = df['yardsGained'] * ~df['any_motion']  # ~df['any_motion'] gives the opposite (False)



    # Group data by offensive formation, receiver alignment, defensive coverage, and defensive type (man/zone)
    grouped_df = df.groupby(['offenseFormation', 'receiverAlignment', 'pff_passCoverage']).agg(
        total_plays=('playId', 'count'),

        avg_yards_gained=('yardsGained', lambda x: x.mean(skipna=True)),   
        success_count=('_playSuccessful', 'sum'),  # it was mean average success rate
        
        pass_freq=('_is_pass', 'sum'), 
        run_freq=('_is_run', 'sum'),

        motion_plays_count=('any_motion', 'sum'),  # Total plays with any motion
        # avg_yards_motion=('yardsGained', lambda x: x[df.loc[x.index, 'any_motion']== True].mean(skipna=True)),  # Avg yards gained with motion
        # avg_yards_no_motion=('yardsGained', lambda x: x[df.loc[x.index, 'any_motion'] == False].mean(skipna=True)),  # Avg yards gained without motion

        # Calculate avg yards when motion, ignoring NA values
        avg_yards_motion=('avg_yards_when_motion', lambda x: x.dropna().mean() if not x.dropna().empty else pd.NA),
        # Calculate avg yards when no motion, ignoring NA values
        avg_yards_no_motion=('avg_yards_when_no_motion', lambda x: x.dropna().mean() if not x.dropna().empty else pd.NA),  # If no data, return NA


        # Use mean for yards when pass and run
        avg_yards_when_pass=('avg_yards_when_pass', 'mean'),
        avg_yards_when_run=('avg_yards_when_run', 'mean')

    ).reset_index()

    grouped_df['pass_rate'] = ((grouped_df['pass_freq']/ grouped_df['total_plays'])*100).round(2)
    grouped_df['run_rate'] =  ((grouped_df['run_freq']/ grouped_df['total_plays'])*100).round(2)
    grouped_df['formation_perent'] = ((grouped_df['total_plays'] / grouped_df.shape[0])*100).round(2)
    grouped_df['success_rate'] = ((grouped_df['success_count']/ grouped_df['total_plays'])*100).round(2)

    # Add motion percentage column
    grouped_df['motion_percentage'] = (grouped_df['motion_plays_count'] / grouped_df['total_plays']) * 100



    if sort_by == 'success_rate':
        # Step 3: Sort the results to highlight defensive weaknesses (e.g., highest avg yards gained)
        results_sorted = grouped_df.sort_values(by=['avg_yards_gained', 'success_rate'], ascending=ascending)
    else: 
        # Sort by motion percentage
        results_sorted = grouped_df.sort_values(by='motion_percentage', ascending=ascending) #, inplace=True)
    
    return results_sorted

"""
Custom aggregation based on list passed 
"""
def playsdf_offense_deffense_custom_agg(df, agg_by_list=[], sort_by='success_rate', ascending=False):
        # Group data by offensive formation, receiver alignment, defensive coverage, and defensive type (man/zone)
    #grouped_df = df.groupby(['offenseFormation', 'receiverAlignment', 'pff_passCoverage']).agg(
    # if not agg param is passed, then agg by both offense/deffense strategies 
    if len(agg_by_list) == 0 or agg_by_list is None: 
        agg_by_list = ['offenseFormation', 'receiverAlignment', 'pff_passCoverage']
    

    grouped_df = df.groupby(agg_by_list).agg(
        total_plays=('playId', 'count'),

        avg_yards_gained=('yardsGained', lambda x: x.mean(skipna=True)),   
        success_count=('_playSuccessful', 'sum'),  # it was mean average success rate
        
        pass_freq=('_is_pass', 'sum'), 
        run_freq=('_is_run', 'sum'),

        motion_plays_count=('any_motion', 'sum'),  # Total plays with any motion
        avg_yards_motion=('yardsGained', lambda x: x[df.loc[x.index, 'any_motion']== True].mean(skipna=True)),  # Avg yards gained with motion
        avg_yards_no_motion=('yardsGained', lambda x: x[df.loc[x.index, 'any_motion'] == False].mean(skipna=True)),  # Avg yards gained without motion

        avg_yards_when_pass=('_is_pass', lambda x: x[df.loc[x.index, '_is_pass']== 1].mean(skipna=True)),  # Avg yards gained with motion
        avg_yards_when_run=('_is_run', lambda x: x[df.loc[x.index, '_is_run'] == 1].mean(skipna=True)),  # Avg yards gained without motion

    ).reset_index()

    grouped_df['pass_rate'] = ((grouped_df['pass_freq']/ grouped_df['total_plays'])*100).round(2)
    grouped_df['run_rate'] =  ((grouped_df['run_freq']/ grouped_df['total_plays'])*100).round(2)
    grouped_df['formation_perent'] = ((grouped_df['total_plays'] / grouped_df.shape[0])*100).round(2)
    grouped_df['success_rate'] = ((grouped_df['success_count']/ grouped_df['total_plays'])*100).round(2)

    # Add motion percentage column
    grouped_df['motion_percentage'] = (grouped_df['motion_plays_count'] / grouped_df['total_plays']) * 100
    # do you need the one below?
    ###grouped_df['motion_effect'] = grouped_df['avg_yards_motion'] - grouped_df['avg_yards_no_motion']


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
def filter_by_offdeff_strategy(df, offenseFormation=None, receiverAlignment=None, pff_passCoverage=None):
        # Filter the data based on offenseFormation, receiverAlignment, passCoverage, and manZone if they are provided
    if offenseFormation is not None:
        df = df[df['offenseFormation'] == offenseFormation.upper()]
    if receiverAlignment is not None:
        df = df[df['receiverAlignment'] == receiverAlignment]

    # Filter defensive strategy if provided
    if pff_passCoverage is not None:
        df = df[df['pff_passCoverage'] == pff_passCoverage]

    return df 

# ---------------------------------
"""
 Generate the title for the analysis
"""
def generate_analysis_title(gameId=None, offensive_team=None, defensive_team=None, 
                            quarter=None, winning_team=None, offenseFormation=None, 
                            receiverAlignment=None, pff_passCoverage=None):
    """
    Generates a dynamic title for the analysis based on the provided parameters.

    Parameters:
    - gameId (str): ID of the game.
    - offensive_team (str): Name of the offensive team.
    - defensive_team (str): Name of the defensive team.
    - quarter (str/int): Quarter of the game.
    - winning_team (str): Name of the winning team.
    - offenseFormation (str): Offensive formation type.
    - receiverAlignment (str): Receiver alignment description.
    - pff_passCoveragenote (str): Notes about pass coverage.

    Returns:
    - str: Generated title.
    """

    title_parts = ["Analysis of Average Yards Gained"]

    # Add dynamic parts
    if offensive_team:
        title_parts.append(f"by {offensive_team} Offense")
    if defensive_team:
        title_parts.append(f"against {defensive_team} Defense")
    if quarter:
        title_parts.append(f"in Q{quarter}")
    if gameId:
        title_parts.append(f"(Game ID: {gameId})")
    if winning_team:
        title_parts.append(f"- Winning Team: {winning_team}")
    if offenseFormation:
        title_parts.append(f" | Formation: {offenseFormation}")
    if receiverAlignment:
        title_parts.append(f" | Receiver Alignment: {receiverAlignment}")
    if pff_passCoverage:
        title_parts.append(f" | Coverage Note: {pff_passCoverage}")

    # Base analysis description
    title_parts.append(
        "| Analysis includes: Play Type (Motion, Pass, Run) and Time Buckets Influence"
    )

    # Combine all parts into a single title
    return " ".join(title_parts)
