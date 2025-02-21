# NFL Strategy Analysis and Prediction Flask Application

## Overview

This Flask application is designed for NFL competition enthusiasts and analysts, providing an interactive way to analyze and predict the effects of various strategies based on pre/post snap behaviors. Users can explore combinations of offensive and defensive parameters, view insightful visualizations, and leverage AI-powered analysis to gain deeper insights into game strategies.

### Pre-Snap to Post-Snap Analysis: Insights, Predictions, and Visualizations

#### Objective
- Provide actionable insights on offensive and defensive strategies.
- Predict post-snap outcomes based on pre-snap behaviors.
- Enable coaches, analysts, and players to understand tendencies through a user-friendly website.

#### Key Components
- Data Analysis and Visualization
- LLM-Driven Descriptive Insights
- Machine Learning for Yardage Prediction
- Game Playback for Historical Comparison



## Features

### Interactive Analysis
Users can select combinations of the following parameters to perform analysis:
- Offensive Team
- Defensive Team
- Offensive Formation
- Receiver Alignment
- Defense Man Zone
- Quarter

The analysis includes:
- Visualizations showing the effect of different parameters on offensive and defensive strategies.
  - **Motion Effect**: Impact of player motion on strategies.
  - **Run/Pass Effect**: Analysis of running vs passing strategies.
  - **Time Effect**: Influence of timing on gameplay.
- Average yardage for a given combination based on data analysis.

### AI-Powered Insights
The application integrates Claude 3.5 Sonnet (LLM) to provide AI-generated insights for aggregated analysis results. This feature helps users understand the implications of different strategies in a concise and meaningful way.

### Yardage Prediction
Users can predict yardage for a specific combination of parameters using the built-in predictive model, trained on historical data.

## Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.10
- pip (Python package manager)
- Virtual environment (optional but recommended)

### Steps
#### Using Python Environment 

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/nfl-analysis-flask-website.git
   cd nfl-analysis-flask-website
   ```

2. Download the data:
   https://drive.google.com/drive/folders/183R02hnbDRAavB5yN8XOPWq5WsDGSpeP?usp=sharing

3. Extract the data:
   ```bash
   tar -xzvf database.sqlite.tar.gz
   tar -xzvf dataset.tar.gz
   ```

4. Create and activate a virtual environment (optional):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

5. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

6. Run the application:
   ```bash
   python app.py
   ```
   The application will be accessible at `http://127.0.0.1:5001/`.

#### Using Docker Container
   ```bash
   docker-compose up
   ```

## Usage
1. Open the application in your web browser.
2. Select desired combinations of parameters from the dropdown menus.
3. Click "Run Analysis" to view:
   - Visualizations.
   - AI insights for aggregated results.
   - Predicted yardage for the selected combination.
   - Select Interactive Data Visualizations to see 2D play.

## Data Analysis via Game 2D Interactive Visualization
   ![Data Analysis via Game 2D Interactive Visualization](static/game/tracking.gif)

## Tech Stack
- **Backend**: Flask
- **Frontend**: HTML, CSS, JavaScript
- **AI Integration**: Claude 3.5 Sonnet (LLM)
- **Data Analysis**: Pandas, NumPy, Matplotlib, Seaborn, Tensorflow, Keras
- **Predictive Model**: Trained using Scikit-learn

## Project Structure
```
.
├── app.py                   # Main Flask application
├── static/                  # Static files (CSS, JS, images)
├── templates/               # HTML templates
├── models_artifacts/        # Predictive models and data processing artifacts
├── notebook/                # Model training notebook
├── dataset/                 # Sample datasets
├── model_artifacts/         # Model Artifacts
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

## Future Enhancements
- Add more detailed visualizations for play-by-play analysis.
- Enable user-uploaded datasets for custom analysis.
- Improve yardage prediction model accuracy with advanced algorithms.
- Expand AI analysis to cover specific game scenarios.

## License
This project is licensed under the [Unlicense](LICENSE).

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for suggestions and improvements.

## Contact
For questions or feedback, reach out to [Tiger Team](mailto:light.email.work@gmail.com).
