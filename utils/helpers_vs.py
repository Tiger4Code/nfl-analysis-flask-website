# import libraries - updated
import pandas as pd 
import numpy as np
from datetime import datetime
import json 
import sys
import os 
import random
import io
import base64

import matplotlib
matplotlib.use('Agg')  # Use a backend suitable for scripts
import matplotlib.pyplot as plt


import seaborn as sns
import pandas as pd

from utils import helpers

CMAP_LIST = [
    'Blues', 'BuGn', 'BuPu', 'GnBu', 'Greens', 'Greys', 'Oranges', 'OrRd',
    'PuBu', 'PuBuGn', 'PuRd', 'Purples', 'RdPu', 'Reds', 'YlGn', 'YlGnBu',
    'YlOrBr', 'YlOrRd', 'cividis', 'inferno', 'magma', 'plasma', 'viridis'
]

COL_VIS_TITLE_MAPPING = {
    # 'all': '',
    'avg_yards_gained': 'Average Yards Gained ', 
    #'formation_count': "Formation Frequency ",
    #### 'formation_perent': "Formation Percentage ",not working 
    # 'pass_freq':  "Pass Frequency ",
    # 'run_freq': "Run Frequency ",
    'pass_rate': "Pass Percentage ",
    'run_rate': "Run Percentage "
}




# Assuming df is already loaded with data
"""
No Longer Used!!
I am using the single one instead : visualize_off_def_effectiveness_one_attr
Visualize two heatmaps next to each other
"""
def visualize_off_def_effectiveness(df):

    grouped_df = helpers.playsdf_offense_deffense_agg(df)
    # Pivot data to create a matrix format for heatmaps
    # For average yards gained
    heatmap_data_avg = grouped_df.pivot_table(
        index=['offenseFormation', 'receiverAlignment'], 
        columns=['pff_passCoverage'], 
        values='avg_yards_gained'
    )
    
    # For frequency count
    heatmap_data_count = grouped_df.pivot_table(
        index=['offenseFormation', 'receiverAlignment'], 
        columns=['pff_passCoverage'], 
        values='formation_perent', #'formation_perent'# 'formation_count'
    )

    # Create subplots for two heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Plot the first heatmap (Average Yards Gained)
    sns.heatmap(heatmap_data_avg, annot=True, ax=axes[0], cmap="YlOrRd", linewidths=0.5, 
                cbar_kws={'label': 'Avg Yards Gained'}, vmin=0, vmax=10)  # Adjust vmax for temperature control
    axes[0].set_title("Avg Yards Gained by Offensive Formation and Receiver Alignment")
    axes[0].set_xlabel("Defensive Coverage and Type")
    axes[0].set_ylabel("Offensive Formation and Receiver Alignment")
    axes[0].tick_params(axis='x', rotation=90)
    axes[0].tick_params(axis='y', rotation=0)



    # Plot the second heatmap (Frequency Count)
    sns.heatmap(heatmap_data_count, annot=True, ax=axes[1], cmap="Blues", linewidths=0.5, 
                cbar_kws={'label': 'Formation Percent'}, vmin=0, vmax=15)  # Adjust vmax for temperature control
    axes[1].set_title("Formation Percent by Offensive Formation and Receiver Alignment")
    axes[1].set_xlabel("Defensive Coverage and Type")
    axes[1].set_ylabel("Offensive Formation and Receiver Alignment")
    axes[1].tick_params(axis='x', rotation=90)
    axes[1].tick_params(axis='y', rotation=0)

    # Adjust layout and display the plot
    plt.tight_layout()
    plt


"""
Show boxplot distribution
"""
def generalized_boxplot(df, x_columns, y_column, agg_func='mean', title=None):
    """
    Creates a generalized boxplot with customizable x-axis columns and aggregation function.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        x_columns (list): List of column names to combine for the x-axis.
        y_column (str): Column name for the y-axis.
        agg_func (str): Aggregation function for grouping ('mean', 'sum', etc.).
        title (str): Title for the plot.
    
    Returns:
        None
    """
    # Combine specified x_columns into a single column
    df['x_combined'] = df[x_columns].astype(str).agg(' | '.join, axis=1)

    # Aggregate data if necessary
    if agg_func:
        df_grouped = df.groupby('x_combined').agg({y_column: agg_func}).reset_index()
    else:
        df_grouped = df
    
    # Create the boxplot
    sns.boxplot(x='x_combined', y=y_column, data=df_grouped)
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for readability
    plt.xlabel(" | ".join(x_columns))   # Label for x-axis
    plt.ylabel(y_column)
    if title:
        plt.title(title)
    plt.tight_layout()  # Adjust layout for better fit
    plt.show()

# Example usage
# generalized_boxplot(
#     df=df,
#     x_columns=['_game_time_bucket', 'formation', 'alignment'],  # Combine these columns for x-axis
#     y_column='yardsGained',
#     agg_func='mean',  # Aggregation function
#     title="Distribution of Yards Gained by Time Bucket, Formation, and Alignment"
# )

# Call the function to visualize the heatmap
#visualize_off_def_effectiveness(df)

"""
data calculated for visualize_off_def_effectiveness
"""
def calculate_off_def_effectiveness(df):


    results_sorted = helpers.playsdf_offense_deffense_agg(df)

    # Display results
    # print("Defensive Weaknesses by Formation and Coverage Data Size:")
    # print(results_sorted.shape)
    # print("Defensive Weaknesses by Formation and Coverage:")
    # print(results_sorted.head(2))
    # print(results_sorted.columns)


#calculate_off_def_effectiveness(df)
# ---------------------------------------

"""
avg_yards_gained
formation_count
formation_perent
pass_rate
run_rate

Pass aggregated data frame by offense/defense strategies 
"""
def visualize_off_def_effectiveness_one_attr(grouped_df, col='avg_yards_gained'):
    col = col if col is not None else 'avg_yards_gained'
    # print(f" col={col}")

    title='Average Yards Gained '
    if col in COL_VIS_TITLE_MAPPING.keys():
        title = COL_VIS_TITLE_MAPPING[col]

    # grouped_df = helpers.playsdf_offense_deffense_agg(df)
    # grouped_df = helpers.playsdf_offense_deffense_custom_agg(df, agg_by_list)


    # Pivot data to create a matrix format for heatmaps
    # For average yards gained
    heatmap_data_avg = grouped_df.pivot_table(
        index=['offenseFormation', 'receiverAlignment'], 
        columns=['pff_passCoverage'], 
        values=col #'avg_yards_gained'
    )
    selected_cols = ['offenseFormation', 'receiverAlignment', 'pff_passCoverage', col]

    # Plot heatmap for average yards gained by formation and coverage
    plt.figure(figsize=(10, 6)) 
    cmap = random.choice(CMAP_LIST) # "YlOrRd"
    sns.heatmap(heatmap_data_avg, annot=True, cmap=cmap, linewidths=.5, fmt=".2f", annot_kws={"size": 7})

    plt.title(f"{title} by Offensive Formation and Defensive Coverage Combination") #Average Yards Gained
    plt.xlabel("Defensive Coverage")
    plt.ylabel("Offensive Formation")
    plt.show()

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()


    return grouped_df[selected_cols]



"""
    Percentage Pie Chart
"""
def plot_metric_percentage_pie(grouped_df, metric, metric_name):
    """
    Plots a pie chart for the selected metric against 'everything else'.

    Parameters:
    - grouped_df: The DataFrame from the aggregation function.
    - metric: The column name of the selected metric (e.g., 'formation_perent', 'run_rate', 'pass_rate').
    - metric_name: A human-readable name for the selected metric (e.g., 'Formation Percentage').
    """
    # Sum the selected metric and the "others"
    selected_metric = grouped_df[metric].sum()
    # print(f"Selected Metric == {selected_metric}")
    others = 100 - selected_metric  # Assuming percentages add to 100
    
    # Data for the pie chart
    data = [selected_metric, others]
    labels = [metric_name, 'Others']

    # Create pie chart
    plt.figure(figsize=(6, 6))
    plt.pie(data, labels=labels, autopct='%1.1f%%', colors=['#66b3ff', '#ff9999'], startangle=140)
    plt.title(f'{metric_name} vs Everything Else')
    plt.show()

# Example usage:
# Assuming 'grouped_df' is the output from your custom aggregation function
# plot_metric_percentage_pie(grouped_df, metric='formation_perent', metric_name='Formation Percentage')


# -------------------------- Start Motion Effects ------------------------------
"""
    Bar Plot: Motion Percentage by Combined Formation
"""
def visualize_barplot_motion_precent_in_offensive_formations(df):
    # Bar Plot: Motion Percentage by Combined Formation
    plt.figure(figsize=(14, 6))
    sns.barplot(
        data=df.sort_values(by='motion_percentage', ascending=False),
        x=df['offenseFormation'] + ' | ' + df['receiverAlignment'],
        y='motion_percentage',
        palette='viridis'
    )
    plt.title('Motion Percentage by Formation', fontsize=14)
    plt.xlabel('Formation (OffenseFormation | ReceiverAlignment)', fontsize=12)
    plt.ylabel('Motion Percentage (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

"""
    Heatmap: Metrics by Combined Formation Groups (Motion Percent/Avg Yards)
"""
def visualize_heatmap_motion_precent_avg_yards_in_offensive_formations(df):
    # Heatmap: Metrics by Combined Formation Groups
    # Create combined formation column
    df['combined_formation'] = (
        df['offenseFormation'] + ' | ' + df['receiverAlignment']
    )

    # Pivot table
    pivot_data = df.pivot_table(
        index='combined_formation',
        values=['motion_percentage', 'avg_yards_motion', 'avg_yards_no_motion']
    )

    # Handle NaN and ensure numeric data
    pivot_data = pivot_data.fillna(0).apply(pd.to_numeric, errors='coerce')

    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        pivot_data,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        cbar_kws={'label': 'Value'},
    )
    plt.title('Metrics Heatmap by Combined Formation Groups', fontsize=14)
    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Formation (OffenseFormation | ReceiverAlignment)', fontsize=12)
    plt.tight_layout()
    plt.show()


def visualize_scatterplot_motion_effects_on_deffensive_formations(df, pff_passCoverage=None):
    # Prepare data for scatter plot with defensive strategy as facet
    df['defensive_strategy'] = (
        df['pff_passCoverage'] 
    )

    if pff_passCoverage:
        # Filter for the specific defensive strategy
        strategy = f"{pff_passCoverage}"
        df = df[df['defensive_strategy'] == strategy]

        # Check if there is data after filtering
        if df.empty:
            print(f"No data available for the defensive strategy: {strategy}")
            return

        # Increase plot size for a single defense strategy
        g = sns.FacetGrid(
            df,
            col='defensive_strategy',
            col_wrap=1,  # Show as a single plot instead of multiple facets
            height=6,  # Larger plot size for single strategy
            sharex=True,
            sharey=True
        )
    else:
        # Faceted Scatter Plot for all defensive strategies
        g = sns.FacetGrid(
            df,
            col='defensive_strategy',
            col_wrap=3,
            height=4,
            sharex=True,
            sharey=True
        )

    # Plot the scatterplot
    g.map_dataframe(
        sns.scatterplot,
        x='avg_yards_motion',
        y='avg_yards_no_motion',
        size='motion_percentage',
        sizes=(50, 300),
        alpha=0.7
    )

    # Add diagonal reference line to each subplot
    for ax in g.axes.flat:
        ax.axline((0, 0), slope=1, color='gray', linestyle='--', alpha=0.8)

    # Adjust titles and labels
    g.set_titles('{col_name}')
    g.set_axis_labels('Avg Yards Gained with Motion', 'Avg Yards Gained without Motion')

    # Adjust the overall title and space
    g.fig.subplots_adjust(top=0.85)  # Reduce the top space slightly to avoid overlap
    if pff_passCoverage:
        g.fig.suptitle(f'Avg Yards Gained with vs Without Motion for {strategy}', fontsize=16)
    else:
        g.fig.suptitle('Avg Yards Gained with vs Without Motion by Defensive Strategy', fontsize=16)

    # Tight layout for axes and adjust the overall figure layout to avoid overlap
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the layout within a rectangle

    plt.show()


"""
## Impact of Motion on Yards Gained by Offensive Formation for a given Defensive Strategy
"""
def visualize_barplot_motion_effect_on_avg_yard_gained_by_offensive_formation(df, pff_passCoverage=None):

    df = helpers.filter_by_offdeff_strategy(df, offenseFormation=None, receiverAlignment=None, pff_passCoverage=pff_passCoverage)
    # If no data is left after filtering, show a message and return
    if df.empty:
        print(f"No data available for the selected filters.")
        return
    
    df['defensive_strategy'] = (
        df['pff_passCoverage'] 
    )
    df['offensive_formation'] = (
        df['offenseFormation'] + ' | ' + df['receiverAlignment']
    )

    # Melt the data for grouped bar plots
    melted_data = df.melt(
        id_vars=['defensive_strategy', 'offensive_formation', 'motion_percentage'],
        value_vars=['avg_yards_motion', 'avg_yards_no_motion'],
        var_name='motion_type',
        value_name='avg_yards'
    )

    # Create separate plots for each defensive strategy
    unique_defensive_strategies = df['defensive_strategy'].unique()

    for defensive_strategy in unique_defensive_strategies:
        # Filter data for the specific defensive strategy
        subset = melted_data[melted_data['defensive_strategy'] == defensive_strategy]
        motion_data = df[df['defensive_strategy'] == defensive_strategy]
        
        # Initialize the plot
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=subset,
            x='offensive_formation',
            y='avg_yards',
            hue='motion_type',
            palette='coolwarm'
        )
        
        # Add title and labels
        plt.title(
            f"Impact of Motion on Yards Gained by Offensive Formation\n(Defensive Strategy: {defensive_strategy})",
            fontsize=14
        )
        plt.xlabel('Offensive Formation', fontsize=12)
        plt.ylabel('Average Yards Gained', fontsize=12)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Highlight motion percentage using color intensity
        bars = plt.gca().patches
        for i, bar in enumerate(bars):
            if i < len(motion_data):  # Match bar count to subset
                color_intensity = motion_data.iloc[i % len(motion_data)]['motion_percentage'] / 100
                bar.set_facecolor((1 - color_intensity, color_intensity, 0.5))
        
        # Show the plot
        plt.tight_layout()
        plt.show()

# visualize_barplot_motion_effect_on_avg_yard_gained_by_offensive_formation(df)
# visualize_barplot_motion_effect_on_avg_yard_gained_by_offensive_formation(df,pff_passCoverage='Cover-3')
# avg_yards_cols_list=[avg_yards_when_pass, avg_yards_when_run], percentage_col=pass_rate #, run_rate
# avg_yards_cols_list=[avg_yards_motion, avg_yards_no_motion], percentage_col=motion_percentage
# avg yards run pass is not working as expected, it is showing equal values; percentage is correct 
def testing_generalization_visualize_barplot_motion_effect_on_yardge_for_defense_foreach_defense_formations(df, 
                                                                                     avg_yards_cols_list,
                                                                                     percentage_col,
                                                                                     percent_axis_title,
                                                                                     offenseFormation=None, 
                                                                                     receiverAlignment=None, 
                                                                                     pff_passCoverage=None,
                                                                                     filterData=False):
    # print(df.head())
    # print(df.columns)
    if filterData:
        df = helpers.filter_by_offdeff_strategy(df, offenseFormation, receiverAlignment, pff_passCoverage)
    # If no data is left after filtering, show a message and return
    if df.empty:
        print(f"No data available for the selected filters.")
        return

    palette = 'pastel'  # Palette for the barplot

    df['offensive_formation'] = (
        df['offenseFormation'] + ' | ' + df['receiverAlignment']
    )

    df['defensive_strategy'] = (
        df['pff_passCoverage'] 
    )

    # cols_list = ['avg_yards_motion', 'avg_yards_no_motion']
    # Melt the data for grouped bar plots
    melted_data = df.melt(
        id_vars=['defensive_strategy', 'offensive_formation', percentage_col],
        value_vars=avg_yards_cols_list, 
        var_name='motion_pass_type',
        value_name='avg_yards'
    )

    # Initialize the plot with primary y-axis
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plotting the barplot on the primary y-axis
    sns.barplot(
        data=melted_data,
        x='defensive_strategy',
        y='avg_yards',
        hue='motion_pass_type',
        palette='coolwarm',
        ax=ax1
    )

    # Customize the primary y-axis
    ax1.set_xlabel('Defensive Strategy', fontsize=12)
    ax1.set_ylabel('Average Yards Gained', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90, ha='center')

    # Create a secondary y-axis for raw 'avg_yards_motion' and 'avg_yards_no_motion' values
    ax2 = ax1.twinx()
    ax2.set_ylabel(f"{percent_axis_title} Percentage", fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Plot motion percentage on the secondary axis
    for col_percent in [percentage_col]: # ['avg_yards_motion', 'avg_yards_no_motion']:
        sns.lineplot(
            data=df,
            x='defensive_strategy',
            y=col_percent,
            label=col_percent,
            ax=ax2,
            marker='o',
            linestyle='--'
        )


    # Annotate the bars with percentage values
    # for bar, perc in zip(ax1.patches, melted_data[percentage_col]):
    #     ax1.annotate(
    #         f"{perc:.2f}%",  # Format the percentage value
    #         (bar.get_x() + bar.get_width() / 2, bar.get_height()),  # Position at the center of the bar
    #         ha='center',  # Horizontal alignment
    #         va='bottom',  # Vertical alignment
    #         fontsize=10,  # Font size
    #         color='black'  # Text color
    #     )


    # Add title
    offform = offenseFormation.upper() if offenseFormation is not None else "?"
    recealig = receiverAlignment.upper() if receiverAlignment is not None else "?"
    plt.title(
        f"Impact of {percent_axis_title} on Yards Gained for Offensive Formation: {offform} | {recealig}",
        fontsize=14
    )

    # Adjust legend and layout
    ax1.legend(loc='upper left')
    #ax2.legend(loc='upper right')
    fig.tight_layout()

    # Show the plot
    plt.show()

    return df 



def visualize_barplot_motion_effect_on_yardge_for_defense_foreach_defense_formations(df, 
                                                                                     offenseFormation=None, 
                                                                                     receiverAlignment=None, 
                                                                                     pff_passCoverage=None,
                                                                                     filterData=False):
    # print(df.head())
    # print(df.columns)
    if filterData:
        df = helpers.filter_by_offdeff_strategy(df, offenseFormation, receiverAlignment, pff_passCoverage)
    # If no data is left after filtering, show a message and return
    if df.empty:
        print(f"No data available for the selected filters.")
        return

    palette = 'pastel'  # Palette for the barplot

    df['offensive_formation'] = (
        df['offenseFormation'] + ' | ' + df['receiverAlignment']
    )

    df['defensive_strategy'] = (
        df['pff_passCoverage'] 
    )

    # Melt the data for grouped bar plots
    melted_data = df.melt(
        id_vars=['defensive_strategy', 'offensive_formation', 'motion_percentage'],
        value_vars=['avg_yards_motion', 'avg_yards_no_motion'],
        var_name='motion_type',
        value_name='avg_yards'
    )

    # Initialize the plot with primary y-axis
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plotting the barplot on the primary y-axis
    sns.barplot(
        data=melted_data,
        x='defensive_strategy',
        y='avg_yards',
        hue='motion_type',
        palette='coolwarm',
        ax=ax1
    )

    # Customize the primary y-axis
    ax1.set_xlabel('Defensive Strategy', fontsize=12)
    ax1.set_ylabel('Average Yards Gained (Aggregated)', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90, ha='center')

    # Create a secondary y-axis for raw 'avg_yards_motion' and 'avg_yards_no_motion' values
    ax2 = ax1.twinx()
    ax2.set_ylabel('Motion Percentage', fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Plot motion percentage on the secondary axis
    for motion_percentage in ['motion_percentage']: # ['avg_yards_motion', 'avg_yards_no_motion']:
        sns.lineplot(
            data=df,
            x='defensive_strategy',
            y=motion_percentage,
            label=motion_percentage,
            ax=ax2,
            marker='o',
            linestyle='--'
        )

    # Add title
    offform = offenseFormation.upper() if offenseFormation is not None else "?"
    recealig = receiverAlignment.upper() if receiverAlignment is not None else "?"
    plt.title(
        f"Impact of Motion on Yards Gained for Offensive Formation: {offform} | {recealig}",
        fontsize=14
    )

    # Adjust legend and layout
    ax1.legend(loc='upper left')
    #ax2.legend(loc='upper right')
    fig.tight_layout()

    # Show the plot
    plt.show()

    return df 


"""
Impact of Motion on Average Yards Gained by Defensive Strategy"""
def xx_visualize_barplot_motion_effect_on_yardge_for_defense_foreach_defense_formations(df, 
                                                                                     offenseFormation=None, 
                                                                                     receiverAlignment=None, 
                                                                                     pff_passCoverage=None,
                                                                                     filterData=False):
    if filterData:
        df = helpers.filter_by_offdeff_strategy(df, offenseFormation, receiverAlignment, pff_passCoverage)
    # If no data is left after filtering, show a message and return
    if df.empty:
        print(f"No data available for the selected filters.")
        return

    palette = 'pastel' # ['#ff9999', '#66b3ff']  # 'pastel'

    df['offensive_formation'] = (
        df['offenseFormation'] + ' | ' + df['receiverAlignment']
    )

    # Create the defensive strategy field for later grouping
    df['defensive_strategy'] = (
        df['pff_passCoverage'] 
    )

    #      grouped_df = helpers.playsdf_offense_deffense_agg(df)

    # Melt the data for grouped bar plots
    melted_data = df.melt(
        id_vars=['defensive_strategy', 'offensive_formation', 'motion_percentage'],
        value_vars=['avg_yards_motion', 'avg_yards_no_motion'],
        var_name='motion_type',
        value_name='avg_yards'
    )

    # If we are looking at a specific offensive formation
    #if offenseFormation and receiverAlignment:
    # Initialize the plot
    plt.figure(figsize=(12, 6))

    # Plotting yardage for each defensive strategy based on the given offensive formation
    sns.barplot(
        data=melted_data,
        x='defensive_strategy',
        y='avg_yards',
        hue='motion_type',
        palette='coolwarm'
    )
    
    # Add title and labels
    offform = offenseFormation.upper() if offenseFormation is not None else "?"
    recealig = receiverAlignment.upper() if receiverAlignment is not None else "?"
    plt.title(
        f"Impact of Motion on Yards Gained for Offensive Formation: {offform} | {recealig}",
        fontsize=14
    )
    
    plt.xlabel('Defensive Strategy', fontsize=12)
    plt.ylabel('Average Yards Gained', fontsize=12)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=90, ha='center')  # Set vertical labels

    # Highlight motion percentage using color intensity
    bars = plt.gca().patches
    for i, bar in enumerate(bars):
        # Match motion percentage to the specific bar
        motion_percentage = melted_data.iloc[i % len(melted_data)]['motion_percentage'] / 100
        color = sns.color_palette('coolwarm', as_cmap=True)(motion_percentage)  # Correct color mapping
        bar.set_facecolor(color)

    # Show the plot
    plt.tight_layout()
    plt.show()

    # else:
    #     # If no specific offensive strategy is passed, show multiple plots for each offensive strategy
    #     unique_offensive_strategies = df['offensive_formation'].unique()

    #     for offensive_strategy in unique_offensive_strategies:
    #         # Filter data for the specific offensive strategy
    #         subset = melted_data[melted_data['offensive_formation'] == offensive_strategy]
            
    #         # Initialize the plot
    #         plt.figure(figsize=(12, 6))

    #         # Plotting yardage for each defensive strategy based on the offensive formation
    #         sns.barplot(
    #             data=subset,
    #             x='defensive_strategy',
    #             y='avg_yards',
    #             hue='motion_type',
    #             palette='coolwarm'
    #         )

    #         # Add title and labels
    #         plt.title(
    #             f"Impact of Motion on Yards Gained for Offensive Formation: {offensive_strategy}",
    #             fontsize=14
    #         )
    #         plt.xlabel('Defensive Strategy', fontsize=12)
    #         plt.ylabel('Average Yards Gained', fontsize=12)

    #         # Rotate x-axis labels for better readability
    #         plt.xticks(rotation=90, ha='center')  # Set vertical labels

    #         # Highlight motion percentage using color intensity
    #         bars = plt.gca().patches
    #         for i, bar in enumerate(bars):
    #             # Match motion percentage to the specific bar
    #             motion_percentage = subset.iloc[i % len(subset)]['motion_percentage'] / 100
    #             color = sns.color_palette('coolwarm', as_cmap=True)(motion_percentage)  # Correct color mapping
    #             bar.set_facecolor(color)

    #         # Show the plot
    #         plt.tight_layout()
    #         plt.show()


#visualize_barplot_motion_effect_on_yardge_for_defense_foreach_defense_formations(df)
#visualize_barplot_motion_effect_on_yardge_for_defense_foreach_defense_formations(df, offenseFormation='Shotgun', receiverAlignment='2x1')


# /-------------------------- End Motion Effects ------------------------------
# -------------------------- Start Time Effects ------------------------------
def visualize_heatmap_time_effect_on_offdeff_strategies(df, 
                                                        offensive_strategies_only=True, 
                                                        offenseFormation=None, 
                                                        receiverAlignment=None, 
                                                        pff_passCoverage=None):
    
    df = helpers.filter_by_offdeff_strategy(df, offenseFormation, receiverAlignment, pff_passCoverage)
    # If no data is left after filtering, show a message and return
    if df.empty:
        print(f"No data available for the selected filters.")
        return
    
    # Combine columns for defensive and offensive strategies
    df['offensive_strategy'] = (
        df['offenseFormation'].astype(str) + ' | ' + df['receiverAlignment'].astype(str)
    )
    df['defensive_strategy'] = (
        df['pff_passCoverage'].astype(str)
    )
    
    # Determine the columns for the groupby based on the flag
    groupby_cols = ['_game_time_bucket', 'offensive_strategy'] if offensive_strategies_only else ['_game_time_bucket', 'offensive_strategy', 'defensive_strategy']

    # Step 4: Calculate play success rates across games with dynamic groupby
    grouped = df.groupby(groupby_cols)['yardsGained'].mean().reset_index()

    # Sort the grouped dataframe by `_game_time_bucket` to show the impact
    sorted_grouped = grouped.sort_values(by=['_game_time_bucket', 'yardsGained'], ascending=[True, False])

    # Pivot table for visualization
    pivot = grouped.pivot(index='_game_time_bucket', columns=groupby_cols[1:], values='yardsGained')

    # Heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot, cmap='coolwarm', annot=True, fmt=".2f")
    plt.title("Success Rate by Offensive Formations and Time Buckets (Aggregated Across Games)")
    plt.ylabel("Time Bucket")
    plt.xlabel("Offensive Formation")

    # Reverse the y-axis
    plt.gca().invert_yaxis()
    plt.show()

    return sorted_grouped

# Example of calling the function with additional filters
# visualize_heatmap_time_effect_on_offdeff_strategies(onegame, offensive_strategies_only=False, offenseFormation="Shotgun", receiverAlignment=None, passCoverage="Cover-3", manZone=None)

# Default case (group by both offensive and defensive strategies)
#visualize_heatmap_time_effect_on_offdeff_strategies(onegame)

# Custom case (group by only offensive strategies)
#visualize_heatmap_time_effect_on_offdeff_strategies(onegame, offensive_strategies_only=True)

# Filtering by offenseFormation and receiverAlignment (example)
# visualize_heatmap_time_effect_on_offdeff_strategies(onegame, offensive_strategies_only=True, offenseFormation="Shotgun", receiverAlignment=None)


# ---------------------------------------------------------------------------------
"""
The `visualize_line_time_effect_on_offdeff_strategies` function visualizes trends in play success rates over time, using line plots to show the impact of different offensive and defensive strategies. 
It allows filtering by specific strategies and compares performance across game time buckets.
"""
def visualize_line_time_effect_on_offdeff_strategies(df, offensive_strategies_only=True, 
                                                     offenseFormation=None, 
                                                     receiverAlignment=None, 
                                                     pff_passCoverage=None, 
                                                     filterData=False):
    
    # print(f"daoffenseFormationta  = {offenseFormation}")
    # print(f"offenseFormation  = {offenseFormation}")
    # print(f"pff_passCoverage  = {pff_passCoverage}")

    if filterData:
        df = helpers.filter_by_offdeff_strategy(df, offenseFormation, receiverAlignment, pff_passCoverage)
    # If no data is left after filtering, show a message and return
    if df.empty:
        print(f"No data available for the selected filters.")
        return
    
    # Combine columns for offensive and defensive strategies
    df['offensive_strategy'] = (
        df['offenseFormation'].astype(str) + ' | ' + df['receiverAlignment'].astype(str)
    )
    df['defensive_strategy'] = (
        df['pff_passCoverage'].astype(str) 
    )
    
    # Determine the columns for the groupby based on the flag
    groupby_cols = ['_game_time_bucket', 'offensive_strategy'] if offensive_strategies_only else ['_game_time_bucket', 'offensive_strategy', 'defensive_strategy']

    # Step 4: Calculate play success rates across games with dynamic groupby
    grouped = df.groupby(groupby_cols)['yardsGained'].mean().reset_index()

    # Find the defensive strategy with the highest average yardsGained for the given offensive strategy
    # best_defensive = grouped.loc[grouped['yardsGained'].idxmax()]
    # # Return the best defensive strategy and its success rate
    # best_defensive['defensive_strategy'], best_defensive['yardsGained']

    # Line plot
    plt.figure(figsize=(12, 8))
    
    # Plot line for each strategy combination
    if offensive_strategies_only:
        sns.lineplot(data=grouped, x='_game_time_bucket', y='yardsGained', hue='offensive_strategy', marker='o')
        plt.title("Success Rate by Offensive Formations and Time Buckets (Trends Over Time)")
        plt.xlabel("Time Bucket")
        plt.ylabel("Average Yards Gained")
    else:
        # Plot line for each combination of offensive and defensive strategy
        sns.lineplot(data=grouped, x='_game_time_bucket', y='yardsGained', hue='offensive_strategy', style='defensive_strategy', markers=True)
        plt.title("Success Rate by Offensive and Defensive Formations and Time Buckets (Trends Over Time)")
        plt.xlabel("Time Bucket")
        plt.ylabel("Average Yards Gained")

    # Add a grid to show time sampling
    plt.grid(visible=True, linestyle='--', linewidth=0.9, alpha=0.5)  # Configure grid lines

    plt.legend(title="Strategy")
    plt.xticks(rotation=45)  # Rotate x-axis labels if needed
    plt.grid(True)
    plt.show()

    return grouped

 
# Example of calling the function with additional filters
# visualize_line_time_effect_on_offdeff_strategies(onegame, offensive_strategies_only=False, offenseFormation="Shotgun", receiverAlignment=None, passCoverage="Cover-1", manZone=None)
# # Default case (only offensive strategies)
# #visualize_line_time_effect_on_offdeff_strategies(onegame, offensive_strategies_only=True, offenseFormation="Shotgun", receiverAlignment="3x1")
# visualize_line_time_effect_on_offdeff_strategies(onegame, offensive_strategies_only=True )
# ---------------------------------------------------------------------------------

# /-------------------------- End Time Effects ------------------------------

def base64_image_encoding():
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    return image_base64