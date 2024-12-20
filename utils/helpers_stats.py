

import pandas as pd 
import scipy.stats as stats

"""
ANOVA (Analysis of Variance):
Compare success rates across time buckets

ANOVA is used when you want to compare the means of more than two groups (in this case, different time buckets). Since you are analyzing the success rate across various time buckets (_game_time_bucket), ANOVA is a good option.

"""
def perform_anova_on_time_buckets(df, group_col='_game_time_bucket', success_measure_col='yardsGained'):
    # Check if there are multiple time buckets
    if df[group_col].nunique() > 1:
        # Group by '_game_time_bucket' and aggregate average yards gained
        #grouped_data = df.groupby('_game_time_bucket')['yardsGained'].mean().reset_index()

        # Get the success rates for each time bucket
        time_bucket_groups = [group[success_measure_col].values for name, group in df.groupby(group_col)]

        # Perform One-Way ANOVA to compare the success rates across time buckets
        f_stat, p_value = stats.f_oneway(*time_bucket_groups)

        print(f"ANOVA Results: F-statistic = {f_stat}, p-value = {p_value}")
        
        if p_value < 0.05:
            print("There is a significant difference in success rates across time buckets (p < 0.05).")
        else:
            print("There is no significant difference in success rates across time buckets (p >= 0.05).")
    else:
        print("Not enough time buckets to perform ANOVA.")

"""
t-test:

A t-test is used if you are comparing the success rate between only two time buckets. If you have more than two time buckets, ANOVA would be the preferred method, but you can use t-tests for pairwise comparisons if needed.
"""
def perform_t_test_between_time_buckets(df, quarter_no1, quarter_no2, success_measure_col='yardsGained'):
    # Perform a t-test between two time buckets
    # group1 = df[df['_game_time_bucket'] == bucket1]['yardsGained']
    # group2 = df[df['_game_time_bucket'] == bucket2]['yardsGained']

    group1 = df[df['quarter'] == quarter_no1][success_measure_col]
    group2 = df[df['quarter'] == quarter_no2][success_measure_col]

    # Perform a t-test
    t_stat, p_value = stats.ttest_ind(group1, group2)
    
    print(f"T-test between quarter: '{quarter_no1}' and quarter: '{quarter_no2}': t-statistic = {t_stat}, p-value = {p_value}")
    
    if p_value < 0.05:
        print(f"There is a significant difference between the success rates of {quarter_no1} and {quarter_no2} (p < 0.05).")
    else:
        print(f"There is no significant difference between the success rates of {quarter_no1} and {quarter_no2} (p >= 0.05).")

# print()
# print(f"----------------- ANOVA -----------------------------------------")
# # Perform ANOVA on time buckets to compare success rates
# perform_anova_on_time_buckets(df)
# print()
# print(f"----------------- T-Test -----------------------------------------")
# # Or, if you want to compare two specific time buckets (e.g., "1st Quarter" vs "4th Quarter")
# perform_t_test_between_time_buckets(df, 1, 2) 
# perform_t_test_between_time_buckets(df, 1, 3) 
# perform_t_test_between_time_buckets(df, 1, 4) 