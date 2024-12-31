from flask import Flask, render_template
from flask import  request, jsonify

# import io
# import os 
# import base64
# import matplotlib.pyplot as plt 
# import time 
# from utils.bedrock.llm_prompts import LLMPrompt
# from utils.bedrock.llm_functions import LLMService

import pandas as pd
from utils import helpers, features_engineering #, helpers_vs#, helpers_stats
from utils import general_helpers as ghelpers



app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['WTF_CSRF_ENABLED'] = True


from flask_wtf.csrf import CSRFProtect
csrf = CSRFProtect(app)



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

    pff_passCoverage_list = df.pff_passCoverage.unique()
    game_clock = features_engineering.generate_game_clock_values()
    # print("----------------- game clock ------------")
    # print(game_clock)
    # print("----------------- downs ------------")
    # print(df.down.unique())
    # print("----------------- game clock ------------")
    # Context to pass to the template
    context = {
        'unique_teams': unique_teams,
        'unique_games': unique_games,
        'unique_quarters': unique_quarters,
        'games': games,
        'game_clock': game_clock,
        'downs': df.down.unique(),
        'yardline_numbers': features_engineering.generate_yardline_numbers(),
        'offense_formations': df.offenseFormation.unique(),
        'receiver_alignments': df.receiverAlignment.unique(),
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
    # print("-------------- size of df Load from cached data -------------")
    # print(df.shape[0])


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
    # print("-------------- size of df Load from filtered_df -------------")
    # print(filtered_df.shape[0])
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
    winning_team = request.form.get("winning_team") if request.form.get("winning_team") else None
    game_id = request.form.get("game") if request.form.get("game") else None
    quarter = request.form.get("quarter") if request.form.get("quarter") else 1
    offense_formation = request.form.get("offenseFormation") if request.form.get("offenseFormation") else None
    receiver_alignment = request.form.get("receiverAlignment") if request.form.get("receiverAlignment") else None
    pff_pass_coverage = request.form.get("pff_passCoverage") if request.form.get("pff_passCoverage") else None

    downs = request.form.get("downs") if request.form.get("downs") else 1
    game_clock = request.form.get("game_clock") if request.form.get("game_clock") else "00:30"
    yardline_number = request.form.get("yardline_number") if request.form.get("yardline_number") else 1

    # "offenseFormation", "receiverAlignment", "pff_passCoverage", 
    # "down", "possessionTeam", "defensiveTeam",
    # "quarter", 'gameClock', 
    # "yardlineNumber" , "playDirection"
    
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
            print("-------------- prediction ---------------")
            print(prediction)
        return jsonify({'prediction': prediction})
    else: 
        return jsonify({"error": "Please select offense, defense teams, their strategies, besides downs, game clock, and yards number"})


   

    return jsonify({"error": "Sorry, no matching data!"})



if __name__ == '__main__':
    app.run(debug=False)

