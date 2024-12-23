from utils import helpers, helpers_vs
import matplotlib.pyplot as plt 

import io
import os 
import base64
import matplotlib.pyplot as plt 
import time 
import pandas as pd
from utils import helpers, helpers_vs#, helpers_stats

from utils.bedrock.llm_prompts import LLMPrompt
from utils.bedrock.llm_functions import LLMService


"""
    Shows 3 visualizations for given data 
    - Gained Yardage Heatmap
    - Time effect 
    - Motion effect 
    - Pass/run effect 

    Returns image_list, motion_df, time_grouped_df

"""
def visualize_data_func(filtered_df, offenseFormation=None, receiverAlignment=None, pff_passCoverage=None):
    image_list = []

    plt.figure(figsize=(12, 8))

    # 0) Calculate aggregated data by offense/defense strategies 
    aggregated_data_by_off_deff_df = helpers.playsdf_offense_deffense_agg(filtered_df, sort_by='motion_percentage')

    # 1) Avg yards, formation percent, passPercent, runPercent, motion percent
    sorted_grouped = helpers_vs.visualize_off_def_effectiveness_one_attr(aggregated_data_by_off_deff_df, col='avg_yards_gained')
    #Save the plot to a BytesIO stream
    image_list.append(helpers_vs.base64_image_encoding())


    # 2) Time Effect 
    time_grouped_df = helpers_vs.visualize_line_time_effect_on_offdeff_strategies(filtered_df, 
                            offenseFormation,  # Example default
                            receiverAlignment,
                            pff_passCoverage)  
    image_list.append(helpers_vs.base64_image_encoding())
    # save_df_to_csv(time_grouped_df, 'time_grouped_df.csv')
            
    # 3) Motion Effect
    # note, int he notebook, fitered_data basically is the aggrated data 
    avg_yards_cols_list=['avg_yards_motion', 'avg_yards_no_motion']
    percentage_col='motion_percentage'
    percent_axis_title="Motion Percentage"
    motion_df = helpers_vs.testing_generalization_visualize_barplot_motion_effect_on_yardge_for_defense_foreach_defense_formations(aggregated_data_by_off_deff_df, 
                                                                                                avg_yards_cols_list, 
                                                                                                percentage_col,
                                                                                                percent_axis_title,
                                                                                                offenseFormation, 
                                                                                                receiverAlignment, 
                                                                                                pff_passCoverage)
    image_list.append(helpers_vs.base64_image_encoding())
    # save_df_to_csv(motion_df, 'motion_df.csv')

    avg_yards_cols_list=['avg_yards_when_pass', 'avg_yards_when_run']
    percentage_col='pass_rate'
    percent_axis_title="Pass"

    pass_df = helpers_vs.testing_generalization_visualize_barplot_motion_effect_on_yardge_for_defense_foreach_defense_formations(aggregated_data_by_off_deff_df, 
                                                                                                avg_yards_cols_list, 
                                                                                                percentage_col,
                                                                                                percent_axis_title,
                                                                                                offenseFormation, 
                                                                                                receiverAlignment, 
                                                                                                pff_passCoverage)
    image_list.append(helpers_vs.base64_image_encoding())
    # save_df_to_csv(pass_df, 'pass_df.csv')



    # Serialize sorted_grouped for table rendering
    # sorted_grouped = sorted_grouped.fillna("") 
    # sorted_grouped_json = sorted_grouped.to_dict(orient='records')
    # time_grouped_df = time_grouped_df.fillna("") 
    # time_grouped_df_json = time_grouped_df.to_dict(orient='records')
    # motion_df = motion_df.fillna("") 
    # motion_df_json = motion_df.to_dict(orient='records')

    return image_list, motion_df, time_grouped_df

# Function to save DataFrame to CSV
def save_df_to_csv(df, filename):
    # Specify the directory to save the CSV file
    folder_path = "csv_local_data"  # You can change this to any folder you prefer
    
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Generate the full file path
    file_path = os.path.join(folder_path, filename)

    # Save the DataFrame to CSV
    df.to_csv(file_path, index=False)
    # (f"CSV saved to {file_path}")
    return file_path  # Return the file path in case you want to use it later


"""
Generate Analysis using GAI for existing matches
"""
def genai_analysis(aggregated_data, time_data, offensive_team=None, defensive_team=None):
    # Start the timer
    start_time = time.time()
    llmservice = LLMService()

    prompt = LLMPrompt.generate_nfl_analysis_prompt(aggregated_data=aggregated_data, time_data=time_data, offensive_team=offensive_team, defensive_team=defensive_team )
      
    # print("---------------- genai_analysis ---------------------")
    # print(prompt)
    # print("-------------------------------------")
    
    answer = llmservice.invoke_model(prompt)
    # print(answer)
    # print("-------------------------------------")

    # End the timer and calculate the elapsed time
    end_time = time.time()
    execution_time = end_time - start_time

    # Print the execution time
    print(f"Execution Time: {execution_time:.4f} seconds")
    return answer


"""
Generate Analysis using GAI for NON-existing matches
"""
def genai_future_analysis(
                        offense_aggregated_df, 
                        offense_time_grouped_df, 
                        defense_aggregated_df, 
                        defense_time_grouped_df, 
                        offensive_team=None, 
                        defensive_team=None):
    # Start the timer
    start_time = time.time()
    llmservice = LLMService()

        
    prompt = LLMPrompt.generate_nfl_future_analysis_prompt(offense_aggregated_df, 
                                                           offense_time_grouped_df, 
                                                           defense_aggregated_df, 
                                                           defense_time_grouped_df, 
                                                           offensive_team=offensive_team, 
                                                           defensive_team=defensive_team )
      
    # print("------------------ genai_future_analysis -------------------")
    # print(prompt)
    # print("-------------------------------------")
    
    answer = llmservice.invoke_model(prompt)
    # print(answer)
    # print("-------------------------------------")

    # End the timer and calculate the elapsed time
    end_time = time.time()
    execution_time = end_time - start_time

    # Print the execution time
    print(f"Execution Time: {execution_time:.4f} seconds")
    return answer

