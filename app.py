from flask import Flask, render_template
from flask import  request, jsonify
from threading import Timer
import sqlite3
import os 
import json
import subprocess
import webbrowser


import pandas as pd
from utils import helpers, features_engineering #, helpers_vs#, helpers_stats
from utils import general_helpers as ghelpers



app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['WTF_CSRF_ENABLED'] = True


from flask_wtf.csrf import CSRFProtect
csrf = CSRFProtect(app)


connection = sqlite3.connect('database.sqlite', check_same_thread=False)
connection.row_factory = sqlite3.Row

AGGREGATED_DATA_TYPE = 1
RAW_DATA_TYPE = 2

# Visualization types
VISUALIZATIONS = {
    1: "Line Plot",
    2: "Heatmap",
    3: "Bar Chart",
    4: "Scatter Plot",
}

# Statistical analysis methods
STATISTICAL_ANALYSIS_METHODS = {
    1: "ANOVA",
    2: "t-Test",
    3: "Chi-Square Test",
    4: "Regression Analysis",
}


from functools import lru_cache

@lru_cache(maxsize=1)
def get_cached_dataframe():
    """Load and cache the dataframe."""
    return helpers.prepare_and_load_df(loadFromFile=True)

def open_browser():
    webbrowser.open("http://localhost:5001")



@app.route('/')
def visualize():
    # Read the CSV file
    # gamedf = pd.read_csv(f"{helpers.BASECDIR}/games.csv")
    gamedf = helpers.read_csv_from_s3(helpers.S3_BUCKET_NAME, f"{helpers.BASECDIR}/games.csv")

    unique_teams = sorted(pd.concat([gamedf['homeTeamAbbr'], gamedf['visitorTeamAbbr']]).unique())
    unique_games = gamedf.gameId.unique()
    games = gamedf.to_dict(orient='records')

    # Use the cached dataframe
    df = get_cached_dataframe()
    unique_quarters = sorted(df['quarter'].unique())

    pff_passCoverage_list = df.pff_passCoverage.dropna().unique()
    game_clock = features_engineering.generate_game_clock_values()
    
    # Context to pass to the template
    context = {
        'unique_teams': unique_teams,
        'unique_games': unique_games,
        'unique_quarters': unique_quarters,
        'games': games,
        'game_clock': game_clock,
        'downs': df.down.unique(),
        'yardline_numbers': features_engineering.generate_yardline_numbers(),
        'offense_formations': df.offenseFormation.dropna().unique(),
        'receiver_alignments': df.receiverAlignment.dropna().unique(),
        'coverages': pff_passCoverage_list,
    }

    # Render the Flask template
    return render_template('admintemplate/nfl/visualize.html', **context)




@app.route('/generate_vis', methods=['POST'])
@csrf.exempt  # Remove this if you want CSRF protection enabled
def generate_vis():
    titles_with_images = {}
    image_list = []

    # Extract parameters from the request
    offensive_team = request.form.get("offensive_team") if request.form.get("offensive_team") else None
    defensive_team = request.form.get("defensive_team") if request.form.get("defensive_team") else None
    winning_team = request.form.get("winning_team") if request.form.get("winning_team") else None
    game_id = request.form.get("game") if request.form.get("game") else None
    quarter = request.form.get("quarter") if request.form.get("quarter") else None
    offense_formation = request.form.get("offenseFormation") if request.form.get("offenseFormation") else None
    receiver_alignment = request.form.get("receiverAlignment") if request.form.get("receiverAlignment") else None
    pff_pass_coverage = request.form.get("pff_passCoverage") if request.form.get("pff_passCoverage") else None


    if offensive_team and defensive_team and offensive_team == defensive_team:
        return jsonify({"error": "Offense team should be different from defense team!"})

    # Use the cached dataframe
    df = get_cached_dataframe()

    # Filter dataframe
    filtered_df = helpers.selectOffenseDeffenseTeams(
        df,
        game_id,
        offensive_team,
        defensive_team,
        quarter,
        winning_team,
        offense_formation,
        receiver_alignment,
        pff_pass_coverage,
    )

    if not filtered_df.empty:
        # print("0) Normal Filtering")
        # Perform analysis
        image_list, motion_df, time_grouped_df = ghelpers.visualize_data_func(
            filtered_df, offense_formation, receiver_alignment, pff_pass_coverage
        )

        # Generate AI analysis
        gen_ai_response = ghelpers.genai_analysis(
            aggregated_data=motion_df,
            time_data=time_grouped_df,
            offensive_team=offensive_team,
            defensive_team=defensive_team,
        )

        # Create analysis title
        analysis_title = helpers.generate_analysis_title(
            game_id, offensive_team, defensive_team, quarter, winning_team, offense_formation, receiver_alignment, pff_pass_coverage
        )
        titles_with_images[analysis_title] = image_list
        return jsonify({"titles_with_images": titles_with_images, 'sorted_df': None, 'gen_ai_response': gen_ai_response})

    elif offensive_team or defensive_team:
        if offensive_team:
            #print(f"2) Offensive Filtering for {offensive_team}")
            offense_filtered_df = helpers.selectOffenseDeffenseTeams(
                df, game_id, offensive_team, None, quarter, winning_team, offense_formation, receiver_alignment, pff_pass_coverage
            )
            if offense_filtered_df.shape[0] == 0:
                return jsonify({"error": "Sorry, no matching data!"})

            offense_image_list, offense_motion_df, offense_time_grouped_df = ghelpers.visualize_data_func(
                offense_filtered_df, offense_formation, receiver_alignment, pff_pass_coverage
            )
            titles_with_images[f"Analysis of {offensive_team} Team's Offensive Strategies"] = offense_image_list

        if defensive_team:
            # print(f"3) Defensive Filtering for {defensive_team}")
            defense_filtered_df = helpers.selectOffenseDeffenseTeams(
                df, game_id, None, defensive_team, quarter, winning_team, offense_formation, receiver_alignment, pff_pass_coverage
            )
            defensive_image_list, defense_motion_df, defense_time_grouped_df = ghelpers.visualize_data_func(
                defense_filtered_df, offense_formation, receiver_alignment, pff_pass_coverage
            )
            titles_with_images[f"Analysis of {defensive_team} Team's Offensive Strategies"] = defensive_image_list

        # Generate AI analysis
        gen_ai_response = ghelpers.genai_future_analysis(
            offense_aggregated_df=offense_motion_df,
            offense_time_grouped_df=offense_time_grouped_df,
            defense_aggregated_df=defense_motion_df,
            defense_time_grouped_df=defense_time_grouped_df,
            offensive_team=offensive_team,
            defensive_team=defensive_team,
        )
        return jsonify({"titles_with_images": titles_with_images, 'sorted_df': None, 'gen_ai_response': gen_ai_response})

    return jsonify({"error": "Sorry, no matching data!"})


@app.route('/predict', methods=['POST'])
@csrf.exempt  
def predict():
    # Extract parameters from the request
    offensive_team = request.form.get("offensive_team") if request.form.get("offensive_team") else None
    defensive_team = request.form.get("defensive_team") if request.form.get("defensive_team") else None
    offense_formation = request.form.get("offenseFormation") if request.form.get("offenseFormation") else None
    receiver_alignment = request.form.get("receiverAlignment") if request.form.get("receiverAlignment") else None
    pff_pass_coverage = request.form.get("pff_passCoverage") if request.form.get("pff_passCoverage") else None

    quarter = request.form.get("quarter") if request.form.get("quarter") else 1

    downs = request.form.get("downs") if request.form.get("downs") else 1
    game_clock = request.form.get("game_clock") if request.form.get("game_clock") else "00:30"
    yardline_number = request.form.get("yardline_number") if request.form.get("yardline_number") else 1

    
    if offensive_team is not None and defensive_team is not None and offense_formation is not None and receiver_alignment is not None and pff_pass_coverage is not None:
        if offensive_team and defensive_team and offensive_team == defensive_team:
            return jsonify({"error": "Offense team should be different from defense team!"})
        else: 
            # call model 
            df = features_engineering.create_dataframe_from_values(offenseFormation=offense_formation, 
                                                              receiverAlignment=receiver_alignment, 
                                                              pff_passCoverage=pff_pass_coverage,
                                                              down=downs,
                                                              possessionTeam=offensive_team,
                                                              defensiveTeam=defensive_team,                                               
                                                              quarter=quarter,
                                                              gameClock=game_clock, 
                                                              yardlineNumber=yardline_number, 
                                                              playDirection="left")
            
            prediction = features_engineering.inference(df=df)
        return jsonify({'prediction': prediction})
    else: 
        return jsonify({"error": "Please select offense, defense teams, their strategies, besides downs, game clock, and yards number"})


   

    return jsonify({"error": "Sorry, no matching data!"})

@app.route('/games')
def home():
    # if os.path.exists("database.sqlite") and os.path.getsize("database.sqlite") < 1 * 1024 ** 3:
    #     os.remove("database.sqlite")

    # if os.path.exists("database.sqlite.tar.gz"):
    #     subprocess.run(["tar", "-xzvf", "database.sqlite.tar.gz"], check=True)
    
    query = "SELECT * FROM games WHERE 1"
    cursor = connection.cursor()
    cursor.execute(query)
    games = cursor.fetchall()
    return render_template('admintemplate/nfl/games.html', games=games)

# games to select play
@app.route('/game/<int:id>')
def game(id):
    
    cursor = connection.cursor()
    query = "SELECT * FROM plays WHERE game_id = ? order by play_number ASC"
    cursor.execute(query, [id])
    plays = cursor.fetchall()
    
    cursor.execute("SELECT * FROM games WHERE id = ?", [id])
    game = cursor.fetchone()
    
    return render_template('admintemplate/nfl/plays.html', plays=plays, game=game)

# games to select play
@app.route('/game/<int:id>/play/<int:play_id>')
def play(id, play_id):
    
    cursor = connection.cursor() 
    
    cursor.execute("SELECT * FROM games WHERE id = ?", [id])
    game = cursor.fetchone()
    
    cursor.execute("SELECT * FROM plays WHERE game_id = ? and play_number = ?", [id, play_id])
    play = cursor.fetchone()
    
    cursor.execute("SELECT * FROM teams WHERE 1")
    teams_data = cursor.fetchall()
    
    cursor.execute("SELECT * FROM play_metrics WHERE game_id = ? and play_id = ?", [id, play_id])
    play_metrics = cursor.fetchall()
    

    cursor.execute("SELECT * FROM players WHERE 1")
    player_details = cursor.fetchall()
    
    cursor.execute("SELECT * FROM player_metrics WHERE game_id = ? and play_id = ?", [id, play_id])
    player_metrics = cursor.fetchall()
    
    
    ordinal = ['0th', '1st', '2nd', '3rd', '4th']
    playerMap = {}
    home_players = []
    visiter_players = []
    teams = {}
    player_colors = [
        "#B27300",  # Dark Orange
        "#00B2B2",  # Dark Cyan
        "#B200B2",  # Dark Magenta
        "#005757",  # Dark Teal
        "#2CA5A5",  # Dark Turquoise
        "#B2593A",  # Dark Coral
        "#A3A3C6",  # Dark Lavender
        "#B28F00",  # Dark Gold
        "#B28C8C",  # Dark Peach
        "#86B200",  # Dark Lime
        "#98102B",  # Dark Crimson
        "#350061",  # Dark Indigo
        "#A354A3",  # Dark Violet
        "#580000",  # Dark Maroon
        "#B28F00",  # Dark Amber
        "#6CB26C",  # Dark Mint
        "#00537A",  # Dark Cerulean
        "#B200B2",  # Dark Fuchsia
        "#599F9F",  # Dark Aquamarine
        "#AE584D",  # Dark Salmon
        "#868686",  # Dark Silver
        "#599F00"   # Dark Chartreuse
    ]
    player_color_map = {}

    
    for player in player_details:
        playerMap[player['id']] = player
    
    color_index = 0
    
    for play_metric in play_metrics:
        if play_metric['name'] == game['home_team']:
            hplayer = dict(playerMap[play_metric['player_id']])
            hplayer['was_running_route'] = play_metric['was_running_route']
            hplayer['route_ran'] = play_metric['route_ran']
            hplayer['color'] = player_colors[color_index]
            player_color_map[play_metric['player_id']] = player_colors[color_index]
            home_players.append(hplayer)
        if play_metric['name'] == game['visitor_team']:
            vplayer = dict(playerMap[play_metric['player_id']])
            vplayer['was_running_route'] = play_metric['was_running_route']
            vplayer['route_ran'] = play_metric['route_ran']
            vplayer['color'] = player_colors[color_index]
            player_color_map[play_metric['player_id']] = player_colors[color_index]
            visiter_players.append(vplayer)
        color_index += 1

    for team in teams_data:
        teams[team['abbreviation']] = team['name']
    
    color = {game['home_team']: 'yellow', game['visitor_team']: 'red'}
    players = {}
    frame_count_ball_rece = 0
    
    for playerMetric in player_metrics:
        if playerMetric['player_id'] in players:
            players[playerMetric['player_id']]['playerDataList'].append(
                {
                    'x': playerMetric['x'],
                    'y': playerMetric['y'],
                    's': playerMetric['s'],
                    'a': playerMetric['a'],
                    'o': playerMetric['o'],
                    'frame_type': playerMetric['frame_type'],
                    'frame_id': playerMetric['frame_id'],
                    'event': playerMetric['event']
                }
            )
            if playerMetric['event'] == 'pass_arrived':
                frame_count_ball_rece = playerMetric['frame_id']
        else:
            players[playerMetric['player_id']] = {
                'player_id': playerMetric['player_id'],
                'name': playerMetric['name'],
                'player_color': player_color_map[playerMetric['player_id']],
                'jersey_number': playerMetric['jersey_number'],
                'club': playerMetric['club'],
                'play_direction': playerMetric['play_direction'],
                'color': color[playerMetric['club']],
                'position': playerMap[playerMetric['player_id']]['position'],
                'currentPointIndex': 0, 'down': play['down'],
                'yards_to_go': play['yards_to_go'],
                'possession_team': play['possession_team'],
                'progress': 0,
                'completed': False,
                'playerDataList': [
                    {
                        'x': playerMetric['x'],
                        'y': playerMetric['y'],
                        's': playerMetric['s'],
                        'a': playerMetric['a'],
                        'o': playerMetric['o'],
                        'frame_type': playerMetric['frame_type'],
                        'frame_id': playerMetric['frame_id'],
                        'event': playerMetric['event']
                    }
                ]
            }
            
    return render_template('admintemplate/nfl/canvas.html', play=play, game=game, player_metrics=player_metrics, teams=teams, play_metrics=play_metrics, player_details=player_details, ordinal=ordinal, visiter_players=visiter_players, home_players=home_players, players=json.dumps(list(players.values())), frame_count_ball_rece=frame_count_ball_rece)

@app.route('/games/prediction/<float:yards>/<offensive>/<receiver>/<coverage>')
def prediction_display(yards, offensive, receiver, coverage):

    cursor = connection.cursor() 

    cursor.execute(
        """
        SELECT * FROM plays 
        WHERE pff_pass_coverage = ? 
          AND offense_formation = ?
          AND receiver_alignment = ? 
        """, 
        (coverage, offensive, receiver)
    )
    plays = cursor.fetchall()
        
    return render_template('admintemplate/nfl/model.html', yards=yards, plays=plays)

if __name__ == '__main__':
    Timer(1, open_browser).start()  # Open the browser after starting the server
    app.run(debug=False, port=5001)

