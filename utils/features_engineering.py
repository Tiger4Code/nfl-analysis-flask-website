from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, RobustScaler #StandardScaler
#from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
#from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
import pickle
from tensorflow.keras.models import load_model

SCALER_PATH = 'scaler.pkl'

MODEL_PATH = 'model_artifacts/tuned_neural_network_model.h5'

DOWNS_SCALER_PATH = 'model_artifacts/downs_scaler.pkl'
TIME_SCALER_PATH = 'model_artifacts/time_scaler.pkl'
DISTANCE_SCALER_PATH = 'model_artifacts/distance_scaler.pkl'

# TEAMS_ABBR = sorted(pd.concat([ df['possessionTeam'], df['defensiveTeam']]).unique())
TEAMS_ABBR = [
  "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", 
  "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC", 
  "LA", "LAC", "LV", "MIA", "MIN", "NE", "NO", "NYG", 
  "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS"
]

"""
Remove rows with target colummn outliers by team 
# Example usage:
# Assuming you have a DataFrame named 'dataset' and the target column is 'yardsGained'
# The 'possessionTeam' is the group column
"""
def remove_outliers_by_team(dataset, target_column, group_column):
    # Define a function to remove outliers from a group
    def remove_outliers_from_group(group):
        # Calculate the first (Q1) and third (Q3) quartiles for each group
        Q1 = group[target_column].quantile(0.25)
        Q3 = group[target_column].quantile(0.75)
        
        # Calculate the Interquartile Range (IQR)
        IQR = Q3 - Q1
        
        # Define the lower and upper bounds for acceptable data
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Filter the group by removing outliers
        return group[(group[target_column] >= lower_bound) & (group[target_column] <= upper_bound)]
    
    # Group the dataset by the 'possessionTeam' column and apply the outlier removal
    cleaned_dataset = dataset.groupby(group_column, group_keys=False).apply(remove_outliers_from_group)
    
    return cleaned_dataset
# cleaned_dataset = remove_outliers_by_team(newdf, 'yardsGained', 'possessionTeam')
# newdf.shape, cleaned_dataset.shape


# DataFrame Selector for selecting specific columns - updated
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names): 
        self.attribute_names = attribute_names
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.attribute_names]
    
# Custom transformer to round numerical values
class RoundValues(BaseEstimator, TransformerMixin):
    def __init__(self, decimals=2):  # Set default precision to 2
        self.decimals = decimals
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.round(X, self.decimals)  # Round the values to the specified decimal places


# Custom transformer to drop rows with missing values
class DropMissingValues(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.dropna()


# Data cleaning 
def df_cleaning(df, columns, inplace=False):
    """
    Clean the DataFrame by dropping rows with missing values in specified columns 
    and removing duplicate rows.

    :param df: DataFrame to clean.
    :param columns: List of column names to check for missing values.
    :param inplace: Boolean value indicating whether to modify the DataFrame in place.
                    If True, the operations will be performed on the original DataFrame. 
                    If False, a new DataFrame will be returned.
    :return: Cleaned DataFrame with rows containing NaN in the specified columns dropped,
             and duplicate rows removed. If inplace=True, returns None.
    
    Steps:
    1. Drop rows with missing values (NaN) in the specified columns.
    2. Drop duplicate rows from the DataFrame.
    """


    if inplace: 
        df.dropna(subset=columns, inplace=inplace)
        df.drop_duplicates(inplace=inplace)
        return None
    else: 
        cleaned_df = df.dropna(subset=columns)
        new_df = cleaned_df.drop_duplicates()
        return new_df

# LabelEncoderPipeline is a custom pipeline step for label encoding predefined categories.
class LabelEncoderPipeline:
    def __init__(self, categories):
        self.categories = categories
        self.mapping = {category: idx for idx, category in enumerate(categories)}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.replace(self.mapping).values.reshape(-1, 1)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


# Feature engineering function and creating prepared numpy input
def create_prepared_numpy_array_multi_scaler(df, load_scaler=False):
    # Create the _time_remaining_in_game feature
    def calculate_time_remaining_in_game(row):
        minutes, seconds = map(int, row['gameClock'].split(':'))
        game_clock_in_minutes = minutes + seconds / 60.0
        return (4 - row['quarter']) * 15 + game_clock_in_minutes

    df['_time_remaining_in_game'] = df.apply(calculate_time_remaining_in_game, axis=1)

    # Create the _distance_to_goal feature
    df['_distance_to_goal'] = df.apply(
        lambda row: row['yardlineNumber'] if row['playDirection'] == 'left' else 100 - row['yardlineNumber'],
        axis=1
    )

    # Numerical pipeline for response_time, path_encoded, hour, day_of_week
    numeric_attributes = ['down', '_time_remaining_in_game', '_distance_to_goal']

    predefined_categories = {
        'possessionTeam': TEAMS_ABBR,  # 32
        'defensiveTeam': TEAMS_ABBR,
        'offenseFormation': ['SHOTGUN', 'SINGLEBACK', 'EMPTY', 'I_FORM', 'PISTOL', 'JUMBO', 'WILDCAT'],  # 7
        'receiverAlignment': ['2x1', '3x1', '2x2', '3x2', '4x1', '1x1', '2x0', '1x0', '3x0', '3x3', '4x2'],  # 11
        'pff_passCoverage': ['Cover-3 Seam', 'Red Zone', 'Cover-3', 'Cover-2', 'Cover-1', 'Cover-6 Right',
                             'Quarters', 'Cover 6-Left', '2-Man', 'Goal Line', 'Cover-0', 'Cover-3 Double Cloud',
                             'Prevent', 'Bracket', 'Cover-3 Cloud Right', 'Cover-3 Cloud Left', 'Cover-1 Double',
                             'Miscellaneous'],  # 18
    }

    downs_scaler = RobustScaler()
    time_scaler = RobustScaler()
    distance_scaler = RobustScaler()
    if load_scaler: 
        downs_scaler = load_standard_scaler_from_path(DOWNS_SCALER_PATH)        
        time_scaler = load_standard_scaler_from_path(TIME_SCALER_PATH)
        distance_scaler = load_standard_scaler_from_path(DISTANCE_SCALER_PATH)

    # Numerical pipeline for scaling
    downs_pipeline = Pipeline([
        ('selector', DataFrameSelector(['down'])),
        ('drop_missing', DropMissingValues()),  # Drop rows with missing values
        ('round', RoundValues(decimals=2)),
       # ('scaler', scaler if scaler else StandardScaler()),   
        ('scaler', downs_scaler), 
    ])

    time_pipeline = Pipeline([
        ('selector', DataFrameSelector(['_time_remaining_in_game'])),
        ('drop_missing', DropMissingValues()),  # Drop rows with missing values
        ('round', RoundValues(decimals=2)),
       # ('scaler', scaler if scaler else StandardScaler()),   
        ('scaler', time_scaler), 
    ])

    distance_pipeline = Pipeline([
        ('selector', DataFrameSelector(['_distance_to_goal'])),
        ('drop_missing', DropMissingValues()),  # Drop rows with missing values
        ('round', RoundValues(decimals=2)),
       # ('scaler', scaler if scaler else StandardScaler()),   
        ('scaler', distance_scaler), 
    ])

    prefixed_prepared_data_cols = ['down', '_time_remaining_in_game', '_distance_to_goal']
    onehot_features = ['possessionTeam', 'defensiveTeam', 'offenseFormation', 'receiverAlignment', 'pff_passCoverage']
    for feature in onehot_features:
        prefixed_prepared_data_cols.extend([f"{feature}_{category}" for category in predefined_categories[feature]])

    # Prepare the list of one-hot encoding pipelines
    onehotencoding_pipelines_list = []
    for feature in onehot_features:
        temp_pipeline = Pipeline([
            ('selector', DataFrameSelector([feature])),
            ('drop_missing', DropMissingValues()),  # Drop rows with missing values
            ('onehot_encoder', OneHotEncoder(categories=[predefined_categories[feature]],
                                             sparse_output=False,
                                             handle_unknown='ignore'))  # One-hot encoding for specific values
        ])
        onehotencoding_pipelines_list.append(temp_pipeline)

    # Full pipeline to combine all pipelines
    full_pipeline = Pipeline(steps=[
        ('union', FeatureUnion(transformer_list=[
            ("downs_pipeline", downs_pipeline),
            ("time_pipeline", time_pipeline),
            ("distance_pipeline", distance_pipeline),
            *[
                (f"onehot_pipeline_{i}", pipeline)
                for i, pipeline in enumerate(onehotencoding_pipelines_list)
            ]
        ]))
    ])

    # Prepare the data using the full pipeline
    prepared_data = full_pipeline.fit_transform(df)

    # Save the scaler to a file for later use (optional)
    if not load_scaler: 
        downdscaler = downs_pipeline.named_steps['scaler']  
        save_standard_scaler(downdscaler, DOWNS_SCALER_PATH) 

        timescaler = time_pipeline.named_steps['scaler']  
        save_standard_scaler(timescaler, TIME_SCALER_PATH)

        distancescaler = distance_pipeline.named_steps['scaler']  
        save_standard_scaler(distancescaler, DISTANCE_SCALER_PATH)

    # Convert to DataFrame for further processing if needed
    prepared_df = pd.DataFrame(prepared_data, columns=prefixed_prepared_data_cols)

    # Return prepared data and the scaler (for future use)
    return prepared_data, prepared_df


def save_standard_scaler(scaler, scaler_path=SCALER_PATH):
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

def load_standard_scaler_from_path(scaler_full_path):
    try:
        with open(scaler_full_path, 'rb') as f:
            scaler = pickle.load(f)
            return scaler
    except FileNotFoundError:
        print(f"Error: The file {scaler_full_path} does not exist.")
        return None
    except Exception as e:
        print(f"Error loading scaler: {e}")
        return None



import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
def evaluate_predictions(test_target, test_predictions):
    # Compute Pearson correlation coefficient and p-value
    correlation, p_value = pearsonr(test_target, test_predictions)

    #print(f"type of correlation={type(correlation)}")
    corr_value = correlation
    if isinstance(correlation, np.ndarray):
        #print("my_array is a numpy array")
        corr_value = correlation[0]
    # else:
    #     print("my_array is 'numpy.float64'")
    #     corr_value = correlation

    
    # Print the correlation coefficient and p-value
    print(f"Pearson Correlation Coefficient: {corr_value:.4f}")
    print(f"P-value: {p_value:.4f}")

    # Interpretation of Pearson Correlation Coefficient
    if correlation > 0.8:
        correlation_interpretation = "Strong positive correlation: The model's predictions are very close to the true values."
    elif 0.5 < correlation <= 0.8:
        correlation_interpretation = "Moderate positive correlation: The model is reasonably accurate in predicting the target values."
    elif 0 < correlation <= 0.5:
        correlation_interpretation = "Weak positive correlation: The model's predictions have some accuracy but could be improved."
    elif correlation == 0:
        correlation_interpretation = "No correlation: The model's predictions are not linearly related to the true values."
    elif -0.5 <= correlation < 0:
        correlation_interpretation = "Weak negative correlation: As the actual values increase, the predicted values decrease slightly."
    elif -0.8 <= correlation < -0.5:
        correlation_interpretation = "Moderate negative correlation: The predictions are inversely related to the true values."
    else:
        correlation_interpretation = "Strong negative correlation: As the actual values increase, the predicted values decrease significantly."

    print(f"Interpretation of Correlation: {correlation_interpretation}")

    # Interpretation of P-value
    if p_value < 0.05:
        p_value_interpretation = "The correlation is statistically significant (p-value < 0.05), indicating the relationship is unlikely to have occurred by chance."
    else:
        p_value_interpretation = "The correlation is not statistically significant (p-value >= 0.05), suggesting the relationship may be due to random chance."

    print(f"Interpretation of P-value: {p_value_interpretation}")

    # Scatter plot to visualize the relationship
    plt.figure(figsize=(10, 6))
    plt.scatter(test_target, test_predictions, color='blue', alpha=0.6, label=f'Correlation: {corr_value:.4f}')
    plt.plot([test_target.min(), test_target.max()], [test_target.min(), test_target.max()], color='red', linestyle='--', label='Perfect Prediction')
    plt.xlabel('Actual Yards Gained')
    plt.ylabel('Predicted Yards Gained')
    plt.title('Scatter Plot of Actual vs Predicted Yards Gained')
    plt.legend()
    plt.show()

    return correlation, p_value


"""
    Given input variables, create a dataframe of these variables 
"""
def create_dataframe_from_values(
    offenseFormation, receiverAlignment, pff_passCoverage, 
    down, possessionTeam, defensiveTeam, quarter, gameClock, 
    yardlineNumber, playDirection
):
    quarter = int(quarter)
    down = int(down)
    yardlineNumber = int(yardlineNumber)
    # Create a dictionary to map column names to their corresponding values
    data = {
        "offenseFormation": [offenseFormation],
        "receiverAlignment": [receiverAlignment],
        "pff_passCoverage": [pff_passCoverage],
        "down": [down],
        "possessionTeam": [possessionTeam],
        "defensiveTeam": [defensiveTeam],
        "quarter": [quarter],
        "gameClock": [gameClock],
        "yardlineNumber": [yardlineNumber],
        "playDirection": [playDirection]
    }
    print('---------data -------')
    print(data)
    # Create a DataFrame from the dictionary
    df = pd.DataFrame(data)
    return df


"""
    Infernece 
"""
def inference(df):
    
    # Load the saved model
    loaded_model = load_model(MODEL_PATH)

    # Data Preparation 
    prepared_data, prepared_df = create_prepared_numpy_array_multi_scaler(df=df, load_scaler=True)
    predictions = loaded_model.predict(prepared_data)
    prediction = round(float(predictions[0][0]), 3)
    return prediction

def generate_game_clock_values():
    # Generate values for hours from 1 to 14
    values = [f"{hour:02}:00" for hour in range(1, 15)]
    values.insert(0, "00:30")
    values.append("14:30")
    return values

def generate_yardline_numbers():
    return list(range(1, 51))