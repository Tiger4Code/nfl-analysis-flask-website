# import re

TEAM_DIC = {
    'ARI': 'Arizona Cardinals',
    'ATL': 'Atlanta Falcons',
    'BAL': 'Baltimore Ravens',
    'BUF': 'Buffalo Bills',
    'CAR': 'Carolina Panthers',
    'CHI': 'Chicago Bears',
    'CIN': 'Cincinnati Bengals',
    'CLE': 'Cleveland Browns',
    'DAL': 'Dallas Cowboys',
    'DEN': 'Denver Broncos',
    'DET': 'Detroit Lions',
    'GB': 'Green Bay Packers',
    'HOU': 'Houston Texans',
    'IND': 'Indianapolis Colts',
    'JAX': 'Jacksonville Jaguars',
    'KC': 'Kansas City Chiefs',
    'LA': 'Los Angeles Rams',
    'LAC': 'Los Angeles Chargers',
    'LV': 'Las Vegas Raiders',
    'MIA': 'Miami Dolphins',
    'MIN': 'Minnesota Vikings',
    'NE': 'New England Patriots',
    'NO': 'New Orleans Saints',
    'NYG': 'New York Giants',
    'NYJ': 'New York Jets',
    'PHI': 'Philadelphia Eagles',
    'PIT': 'Pittsburgh Steelers',
    'SEA': 'Seattle Seahawks',
    'SF': 'San Francisco 49ers',
    'TB': 'Tampa Bay Buccaneers',
    'TEN': 'Tennessee Titans',
    'WAS': 'Washington Commanders'
}


class LLMPrompt:

    @staticmethod
    def generate_nfl_analysis_prompt(aggregated_data: str, time_data: str, offensive_team: str = None, defensive_team: str = None) -> str:
        """Generate a prompt to analyze NFL data and evaluate offensive and defensive strategies."""

        offensive_team_name = TEAM_DIC.get(offensive_team, None)
        defensive_team_name = TEAM_DIC.get(defensive_team, None)

        offensive_team_analysis_prefix = (
            f"Note, the analysis should be customized for the offensive team '{offensive_team_name}', including how to imporve their chances of success. " if offensive_team_name else ""
        )  # **Added logic to customize the prompt based on the offensive team name**

        deffensive_team_analysis_prefix = (
            f"Note, the analysis should be customized for the offensive team'{defensive_team_name}', including how to imporve their chances of success. " if defensive_team_name else ""
        )  # **Added logic to customize the prompt based on the offensive team name**

        teams_analysis = ""
        if offensive_team_name and defensive_team_name:
            teams_analysis = f"Include a comparative analysis of the offensive strategies of '{offensive_team_name}' against the defensive strategies of '{defensive_team_name}'."
        
        html_format = LLMPrompt.generate_html_format()  # Get the HTML formatting details

        return (
            f"You are an assistant tasked with analyzing NFL game data to identify strategies that result in the highest yardage gained. "

            f"{offensive_team_analysis_prefix}"  
            f"{deffensive_team_analysis_prefix}"  
            f"{teams_analysis}"  

            f"The data is provided in two CSV files. Your task is to analyze the provided data and derive insights on the most effective strategies for both offense and defense. Consider the following criteria:\n\n"

            f"**Main Data Analysis (Aggregated Data)**:\n"
            f"The first Aggregated CSV file is grouped by the columns 'offenseFormation', 'receiverAlignment', and 'pff_passCoverage'. Your analysis should include:\n\n"

            f"1. **Offensive Strategy**:\n"
            f"   - Analyze the formations ('offenseFormation') and receiver alignments ('receiverAlignment') to identify which combinations lead to the highest yardage gained.\n"
            f"   - Assess the effectiveness of passing versus running strategies using the following metrics:\n"
            f"     - 'pass_freq': Total number of pass plays.\n"
            f"     - 'run_freq': Total number of run plays.\n"
            f"     - 'avg_yards_when_pass': Average yards gained when the play involves a pass.\n"
            f"     - 'avg_yards_when_run': Average yards gained when the play involves a run.\n"
            f"     - 'pass_rate': Percentage of plays that are passes.\n"
            f"     - 'run_rate': Percentage of plays that are runs.\n\n"

            f"2. **Defensive Strategy**:\n"
            f"   - Use the 'pff_passCoverage' column to analyze defensive coverage schemes and their effectiveness in limiting yardage.\n\n"

            f"3. **Motion Analysis**:\n"
            f"   - Evaluate the impact of pre-snap motion on play success using the following metrics:\n"
            f"     - 'motion_plays_count': Total number of plays with motion.\n"
            f"     - 'avg_yards_motion': Average yards gained in plays with motion.\n"
            f"     - 'avg_yards_no_motion': Average yards gained in plays without motion.\n"
            f"     - 'motion_percentage': Percentage of plays with motion.\n\n"

            f"4. **Play Volume and Yards**:\n"
            f"   - Consider overall play volume ('total_plays') and average yardage ('avg_yards_gained') across all strategies.\n\n"

            f"**Time-Based Analysis (Time Data)**:\n"
            f"The second CSV file, 'time_data', contains data grouped by '_game_time_bucket', 'offensive_strategy', and 'defensive_strategy'. "
            f"This file should be analyzed to determine how time impacts offensive effectiveness. Consider the following:\n"
            f"   - Analyze the average yards gained ('yardsGained') in relation to the time of the game ('_game_time_bucket'), "
            f"     grouped by offensive and defensive strategies.\n"
            f"   - Identify the time buckets where offensive strategies are most effective.\n\n"

            f"**Deliverables**:\n"
            f"   - A detailed summary of the most effective offensive strategies, including the influence of formations, alignments, and passing versus running.\n"
            f"   - Insights into defensive effectiveness based on coverage schemes.\n"
            f"   - Observations on the role of motion in determining play success.\n"
            f"   - Analysis of the time of the game and its impact on offensive strategy effectiveness.\n"
            f"   - Relevant statistics and trends from both datasets.\n\n"

            f"Main CSV Data:\n{aggregated_data}\n\n"
            f"Time Data CSV:\n{time_data}\n\n"

            f"Return a concise analysis with actionable insights based on the above criteria. Include supporting statistics wherever possible."

            f"{html_format}"

        )


    """
        Used to predict analysis for games that did not take place in the past
    """
    @staticmethod
    def generate_nfl_raw_play_analysis_prompt(offense_filtered_df: str, defense_filtered_df: str, offensive_team: str = None, defensive_team: str = None) -> str:
        """Generate a prompt to analyze raw NFL play data and evaluate offensive and defensive strategies."""

        offensive_team_name = TEAM_DIC.get(offensive_team, None)
        defensive_team_name = TEAM_DIC.get(defensive_team, None)

        offensive_team_analysis_prefix = (
            f"Note, the analysis should be customized for the offensive team '{offensive_team_name}', including how to improve their chances of success. " if offensive_team_name else ""
        )

        defensive_team_analysis_prefix = (
            f"Note, the analysis should be customized for the defensive team '{defensive_team_name}', including how to improve their chances of success. " if defensive_team_name else ""
        )

        teams_analysis = ""
        if offensive_team_name and defensive_team_name:
            teams_analysis = f"Include a comparative analysis of the offensive strategies of '{offensive_team_name}' against the defensive strategies of '{defensive_team_name}'."


        html_format = LLMPrompt.generate_html_format()  # Get the HTML formatting details

        return (
            f"You are an assistant tasked with analyzing raw NFL play data to identify strategies that result in the highest yardage gained. "

            f"{offensive_team_analysis_prefix}"  
            f"{defensive_team_analysis_prefix}"  
            f"{teams_analysis}"  

            f"The data is provided in two raw dataframes. Your task is to analyze the provided data and derive insights on the most effective strategies for both offense and defense. Consider the following criteria:\n\n"

            f"**Offensive Strategy (Offensive Dataframe)**:\n"
            f"The first dataframe, 'offense_filtered_df', contains raw offensive plays data. Your analysis should include:\n\n"
            f"1. **Offensive Play Analysis**:\n"
            f"   - Analyze the offensive play data to identify which formations and alignments lead to the highest yardage gained.\n"
            f"   - Assess the effectiveness of passing versus running strategies using the following metrics:\n"
            f"     - 'pass_freq': Number of pass plays.\n"
            f"     - 'run_freq': Number of run plays.\n"
            f"     - 'avg_yards_when_pass': Average yards gained when the play involves a pass.\n"
            f"     - 'avg_yards_when_run': Average yards gained when the play involves a run.\n"
            f"     - 'pass_rate': Percentage of plays that are passes.\n"
            f"     - 'run_rate': Percentage of plays that are runs.\n\n"

            f"2. **Motion and Pre-snap Movement**:\n"
            f"   - Analyze the impact of pre-snap motion on play success. Use the following metrics:\n"
            f"     - 'motion_plays_count': Number of plays with motion.\n"
            f"     - 'avg_yards_motion': Average yards gained in plays with motion.\n"
            f"     - 'avg_yards_no_motion': Average yards gained in plays without motion.\n"
            f"     - 'motion_percentage': Percentage of plays with motion.\n\n"

            f"**Defensive Strategy (Defensive Dataframe)**:\n"
            f"The second dataframe, 'defense_filtered_df', contains raw defensive plays data. Your analysis should include:\n\n"
            f"1. **Defensive Play Analysis**:\n"
            f"   - Analyze the defensive play data to identify defensive strategies that effectively limit yardage.\n"
            f"   - Assess the effectiveness of defensive formations and coverage schemes using the following metrics:\n"
            f"     - 'coverage_freq': Number of pass coverage plays.\n"
            f"     - 'run_stop_freq': Number of run-stopping plays.\n"
            f"     - 'avg_yards_allowed_pass': Average yards allowed on passing plays.\n"
            f"     - 'avg_yards_allowed_run': Average yards allowed on running plays.\n"
            f"     - 'coverage_success_rate': Percentage of successful pass coverage plays.\n"
            f"     - 'run_stop_success_rate': Percentage of successful run-stopping plays.\n\n"

            f"2. **Motion Defense**:\n"
            f"   - Evaluate the defense's response to motion. Use the following metrics:\n"
            f"     - 'motion_defense_plays_count': Number of defensive plays against motion.\n"
            f"     - 'avg_yards_allowed_motion': Average yards allowed in plays with motion.\n"
            f"     - 'avg_yards_allowed_no_motion': Average yards allowed in plays without motion.\n"
            f"     - 'motion_defense_percentage': Percentage of plays with motion that were defended successfully.\n\n"

            f"**Play Volume and Effectiveness**:\n"
            f"Consider overall play volume in both offensive and defensive plays. Analyze the number of total plays ('total_plays') and average yards gained ('avg_yards_gained') for offense, as well as the average yards allowed for defense.\n\n"

            f"**Deliverables**:\n"
            f"   - A detailed summary of the most effective offensive strategies, including the influence of formations, alignments, and passing versus running.\n"
            f"   - Insights into defensive effectiveness based on coverage schemes.\n"
            f"   - Observations on the role of motion in determining play success.\n"
            f"   - Analysis of the volume of plays and its impact on yardage gained and allowed.\n"
            f"   - Relevant statistics and trends from both dataframes.\n\n"

            f"Offensive Raw Play Data:\n{offense_filtered_df}\n\n"
            f"Defensive Raw Play Data:\n{defense_filtered_df}\n\n"

            f"Return a concise analysis with actionable insights based on the above criteria. Include supporting statistics wherever possible."

            f"{html_format}"
        )


    @staticmethod
    def generate_nfl_future_analysis_prompt(
        offense_aggregated_df, offense_time_grouped_df, 
        defense_aggregated_df, defense_time_grouped_df, 
        offensive_team=None, defensive_team=None
    ) -> str:
        """Generate a prompt to analyze NFL data with motion and time-based groupings."""
        
        offensive_team_name = TEAM_DIC.get(offensive_team, None)
        defensive_team_name = TEAM_DIC.get(defensive_team, None)

        offensive_team_analysis_prefix = (
            f"Note, the analysis should be customized for the offensive team '{offensive_team_name}', including how to improve their chances of success. " 
            if offensive_team_name else ""
        )

        defensive_team_analysis_prefix = (
            f"Note, the analysis should be customized for the defensive team '{defensive_team_name}', including how to improve their chances of success. " 
            if defensive_team_name else ""
        )

        teams_analysis = ""
        if offensive_team_name and defensive_team_name:
            teams_analysis = f"Include a comparative analysis of the offensive strategies of '{offensive_team_name}' against the defensive strategies of '{defensive_team_name}'."

        html_format = LLMPrompt.generate_html_format()  # Get the HTML formatting details

        return (
            f"You are an assistant tasked with analyzing NFL game data, focusing on motion and time-grouped play data to identify strategies that result in the highest yardage gained. "
            
            f"{offensive_team_analysis_prefix}"
            f"{defensive_team_analysis_prefix}"
            f"{teams_analysis}"
            
            f"The data is provided in four CSV files: offensive motion data, offensive time-grouped data, defensive motion data, and defensive time-grouped data. Your task is to analyze the provided data and derive insights on the most effective strategies for both offense and defense. Consider the following criteria:\n\n"

            f"**Offensive Motion and Time Analysis (Offense Data)**:\n"
            f"   - Analyze the motion-related plays in the offensive dataset using the following metrics:\n"
            f"     - 'motion_plays_count': Total number of offensive plays with motion.\n"
            f"     - 'avg_yards_motion': Average yards gained in offensive plays with motion.\n"
            f"     - 'avg_yards_no_motion': Average yards gained in offensive plays without motion.\n"
            f"     - 'motion_percentage': Percentage of offensive plays with motion.\n\n"
            
            f"   - Use the offensive time-grouped dataset to analyze the effectiveness of different offensive strategies over time:\n"
            f"     - 'time_bucket': The period of the game the plays fall under (e.g., 1st quarter, 2nd quarter, etc.).\n"
            f"     - 'avg_yards_gained': Average yardage gained per time bucket.\n"
            f"     - 'pass_run_ratio': The ratio of passing to running plays during specific time periods.\n\n"

            f"**Defensive Motion and Time Analysis (Defense Data)**:\n"
            f"   - Analyze the motion-related plays in the defensive dataset using the following metrics:\n"
            f"     - 'motion_plays_count': Total number of defensive plays with motion.\n"
            f"     - 'avg_yards_motion': Average yards allowed by the defense during plays with motion.\n"
            f"     - 'avg_yards_no_motion': Average yards allowed by the defense during plays without motion.\n"
            f"     - 'motion_percentage': Percentage of defensive plays with motion.\n\n"
            
            f"   - Use the defensive time-grouped dataset to analyze defensive effectiveness over time:\n"
            f"     - 'time_bucket': The period of the game the plays fall under (e.g., 1st quarter, 2nd quarter, etc.).\n"
            f"     - 'avg_yards_allowed': Average yards allowed per time bucket.\n"
            f"     - 'defensive_strategy': The type of defensive strategy used (e.g., man-to-man, zone coverage).\n\n"

            f"**Deliverables**:\n"
            f"   - A detailed summary of the most effective offensive strategies based on motion and time analysis, including insights into motion plays and time-based trends.\n"
            f"   - Insights into defensive effectiveness, including how motion and time influence defensive strategies.\n"
            f"   - Observations on how the effectiveness of both offense and defense evolves over the course of a game.\n"
            f"   - Relevant statistics and trends from both datasets.\n\n"

            f"Offensive Motion Data:\n{offense_aggregated_df}\n\n"
            f"Offensive Time Data:\n{offense_time_grouped_df}\n\n"
            f"Defensive Motion Data:\n{defense_aggregated_df}\n\n"
            f"Defensive Time Data:\n{defense_time_grouped_df}\n\n"
            
            f"{html_format}"
        )


    def generate_html_format() -> str:
        return (
            f"Return the analysis in valid HTML format. The output must adhere to the following requirements:"
            f"    - Use HTML tags for all content (headlines, paragraphs, lists, etc.)."
            f"    - Use `<h2>` for the main title, `<h3>` for section titles, and `<h4>` for subsections."
            f"    - Headlines should have a style attribute to set the color to dark blue (`#00008B`)."
            f"    - Use `<p>` for paragraphs, and `<ol>` or `<ul>` for lists, ensuring proper indentation and structure."
            f"    - Avoid using plain text dashes or any non-HTML formatting; instead, use proper HTML tags (e.g., `<li>` for list items)."
            f"    - Include statistics and actionable insights as part of the HTML structure."
            f"    - Ensure the output is complete, well-formed, and ready to render in a web page."

            f"  Example structure:"
            f"    <h2 style='color: #15384B;'>Main Title</h2>"
            f"    <h3 style='color: #CD1354;'>Section Title</h3>"
            f"    <h4 style='color: #00008B;'>Subsection Title</h4>"
            f"    <p>Content goes here, including statistics and insights.</p>"
            f"    <ol>"
            f"    <li>Actionable insight 1</li>"
            f"    <li>Actionable insight 2</li>"
            f"    </ol> "
        )


    @staticmethod
    def generate_simple_analysis_prompt(team_name: str) -> str:
        """Generate a prompt to analyze the given NFL team based on the team name."""
        
        return (
            f"Analyze the NFL team '{team_name}' and provide a detailed analysis of its strengths and weaknesses. "
            f"Take into account factors such as performance, strengths, and key achievements. "
            f"Please explain them briefly."
        )