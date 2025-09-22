# Research Projects

This directory contains various research projects and analyses. Each project is organized in its own subdirectory with clear documentation and structure following a standardized methodology.

## Research Summary

### 0. Executive Summary
**Data**: [Asset] [Frequency] data from [Start Date] to [End Date] (local Google Shared Drive database)  
**Model**: [Model Type] with [Optimization Method] ([Number] hyperparameter combinations)  
**Target**: [Target Variable] - [Time Period] forward [Prediction Type]  
**Features**: [Number] technical indicators ([List key indicators]) with [Number]-period lags  
**Training Scheme**: Chronological [Train]%/[Val]%/[Test]% train/validation/test split  
**Trading Strategy**: [Strategy Type] with [Risk Management Approach]  
**Results**: [Performance Metrics] and [Trading Results Summary]  

## Project Documentation Template

Each research project should include a comprehensive analysis script (Python/Jupyter) that clearly demonstrates the following components:

### 1. Data Loading and Specification
- **Data Source**: Local Google Shared Drive database connection
- **Data Specifications**:
  - **Stock/Asset**: Specify the exact ticker symbol or asset name
  - **Time Range**: Start and end dates for the analysis period
  - **Data Type**: OHLC, OHLCV, or other data formats
  - **Frequency**: 1-minute bars, 5-minute bars, daily, etc.
- **Data Quality**: Missing value handling, data validation, and preprocessing steps

### 2. Feature Pipeline and Target Definition
- **QuantLib Integration**: Use `quantlib/var_pipeline` for feature engineering
- **Feature Engineering**:
  - **Lookback Variables**: Technical indicators (MACD, RSI, Stochastic, ADX, ATR, MFI, etc.)
  - **Lookforward Variables**: Target variables (Profit Factor, Returns, etc.)
  - **Feature Dimensions**: Number of features and their transformations
- **Target Variable**: Clear definition of prediction target
- **Summary Statistics**:
  - **Distribution Analysis**: Histograms, density plots
  - **Descriptive Statistics**: Mean, standard deviation, skewness, kurtosis
  - **Quantile Analysis**: Percentiles and extreme values
  - **Entropy Measures**: Information content analysis
  - **Correlation Analysis**: Feature correlation heatmaps and matrices

### 3. Additional Data Processing
- **Extra Analysis Requirements**: Any specialized data transformations
- **Processing Steps**: Detailed explanation of additional preprocessing
- **Data Validation**: Quality checks and outlier handling
- **Feature Selection**: Dimensionality reduction or feature importance analysis

### 4. Train/Validation/Test Scheme
- **Temporal Strategy**: 
  - **Single Cut**: Fixed train/validation/test splits
  - **Rolling Window**: Moving window approach
  - **Expanding Window**: Growing training set approach
- **Time Index Specification**:
  - **Training Period**: Exact start and end timestamps
  - **Validation Period**: Exact start and end timestamps  
  - **Testing Period**: Exact start and end timestamps
- **Temporal Integrity**: Chronological ordering to prevent look-ahead bias
- **Timestamp Usage**: How timestamps are used for data subsetting

### 5. Model Training and Validation Details
- **Feature and Target Dimensions**:
  - **Feature Matrix**: Shape and characteristics
  - **Target Vector**: Shape and distribution
- **Model Architecture**:
  - **Model Type**: Boosting Trees (LightGBM, XGBoost), Deep Neural Networks, etc.
  - **Parameter Dimensions**: Number of model parameters
  - **Hyperparameter Dimensions**: Number of hyperparameters to optimize
- **Hyperparameter Optimization**:
  - **Search Strategy**: Grid search, random search, Bayesian optimization
  - **Parameter Ranges**: Specific ranges for each hyperparameter
  - **Best Selection**: Criteria for selecting optimal hyperparameters
- **Training Control**:
  - **Early Stopping**: Validation-based stopping criteria
  - **Stopping Metric**: RMSE, MAE, or other validation metrics
  - **Patience**: Number of epochs/iterations without improvement
- **Evaluation Metrics**:
  - **Regression**: RMSE, MAE, R², MAPE
  - **Classification**: Accuracy, Precision, Recall, F1-Score
  - **Financial**: Sharpe Ratio, Maximum Drawdown, Win Rate

### 6. Results Table
Comprehensive results table showing performance across all train/validation/test periods:

#### Single Cut Approach:
| Dataset | Time Period | RMSE | MAE | R² | MAPE | Sharpe | Max DD | Win Rate |
|---------|-------------|------|-----|----|----- |--------|--------|----------|
| Training | [Start] - [End] | X.XX | X.XX | X.XX | X.XX | X.XX | X.XX | X.XX |
| Validation | [Start] - [End] | X.XX | X.XX | X.XX | X.XX | X.XX | X.XX | X.XX |
| Testing | [Start] - [End] | X.XX | X.XX | X.XX | X.XX | X.XX | X.XX | X.XX |

#### Rolling Window Approach:
| Window | Train Period | Val Period | Test Period | Train RMSE | Val RMSE | Test RMSE | Train MAE | Val MAE | Test MAE | Train R² | Val R² | Test R² |
|--------|--------------|------------|-------------|------------|----------|-----------|-----------|---------|----------|----------|--------|---------|
| 1 | [Start] - [End] | [Start] - [End] | [Start] - [End] | X.XX | X.XX | X.XX | X.XX | X.XX | X.XX | X.XX | X.XX | X.XX |
| 2 | [Start] - [End] | [Start] - [End] | [Start] - [End] | X.XX | X.XX | X.XX | X.XX | X.XX | X.XX | X.XX | X.XX | X.XX |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| N | [Start] - [End] | [Start] - [End] | [Start] - [End] | X.XX | X.XX | X.XX | X.XX | X.XX | X.XX | X.XX | X.XX | X.XX |

#### Rolling Window Summary Statistics:
| Metric | Train Mean | Train Std | Val Mean | Val Std | Test Mean | Test Std |
|--------|------------|-----------|----------|---------|-----------|----------|
| RMSE | X.XX | X.XX | X.XX | X.XX | X.XX | X.XX |
| MAE | X.XX | X.XX | X.XX | X.XX | X.XX | X.XX |
| R² | X.XX | X.XX | X.XX | X.XX | X.XX | X.XX |
| Sharpe | X.XX | X.XX | X.XX | X.XX | X.XX | X.XX |
| Max DD | X.XX | X.XX | X.XX | X.XX | X.XX | X.XX |
| Win Rate | X.XX | X.XX | X.XX | X.XX | X.XX | X.XX |

### 7. Trading Strategy Implementation
- **Signal Generation**: How predictions are converted to trading signals
- **Entry Rules**: Specific conditions for trade entry
- **Position Sizing**: How position sizes are determined
- **Risk Management**:
  - **Take Profit**: Profit-taking levels and rules
  - **Stop Loss**: Loss-cutting levels and rules
  - **Exit Conditions**: When to exit positions
- **Performance Analysis**:
  - **Trade Statistics**: Number of trades, win rate, average win/loss
  - **Return Distribution**: Histogram of trade returns
  - **Cumulative Returns**: Portfolio value over time
  - **Risk Metrics**: Volatility, maximum drawdown, Sharpe ratio

### 8. Analysis and Implications
- **Model Insights**: What the model learned about the data
- **Feature Importance**: Which variables are most predictive
- **Trading Strategy Evaluation**: Strengths and weaknesses
- **Risk Assessment**: Potential risks and limitations
- **Future Improvements**: Suggested enhancements and extensions
