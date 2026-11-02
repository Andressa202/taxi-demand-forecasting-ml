# 🚕 Taxi Demand Forecasting | Time Series & Machine Learning

### Predicting Hourly Taxi Orders Using Historical Demand and Machine Learning

**Python • Pandas • NumPy • Scikit-learn • LightGBM • Time Series Analysis • Feature Engineering**

## 📌 Project Overview

Accurately forecasting taxi demand helps transportation companies plan driver availability, reduce waiting times, and improve operational efficiency.

This project was developed for **Sweet Lift Taxi**, a fictional taxi company that wants to predict the number of taxi orders for the next hour.

Using historical taxi order data, the project explores time series patterns, creates time-based features, and compares four machine learning models.

**Business Goal:** Build a forecasting model capable of achieving a Root Mean Squared Error (RMSE) of no more than **48 taxi orders**.

## 🎯 Project Objectives

- Explore historical taxi order data.
- Resample observations into hourly intervals.
- Identify trends and seasonal patterns.
- Create time-based and lagged features.
- Prevent data leakage during feature engineering.
- Split the dataset chronologically.
- Train and compare regression models.
- Evaluate prediction quality using RMSE.
- Select a model that meets the business requirement.
- Analyze forecasting errors and compare actual versus predicted demand.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Data analysis and modeling |
| Pandas | Time series manipulation and feature engineering |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Scikit-learn | Regression models and evaluation |
| LightGBM | Gradient boosting regression |
| Statsmodels | Time series decomposition |
| Jupyter Notebook | Interactive analysis and documentation |

## 📂 Dataset Description

The project uses `taxi.csv`, containing historical taxi orders recorded at 10-minute intervals.

| Variable | Description |
|---|---|
| `datetime` | Timestamp of the observation |
| `num_orders` | Number of taxi orders |

The target variable is `num_orders`, representing the number of orders during the next hourly interval.

## 🧹 Data Preparation

### 1. Loading and Sorting Data

The dataset was loaded using Pandas, with timestamps parsed as datetime values and used as the DataFrame index.

Observations were sorted chronologically.

### 2. Hourly Resampling

The original 10-minute observations were aggregated into hourly intervals:

```python
data = data.resample('1H').sum()
```

This transformation produced the hourly demand series required for forecasting.

### 3. Data Quality Checks

The analysis included:

- Dataset structure inspection.
- Missing-value verification.
- Chronological ordering.
- Examination of the hourly order series.

## 📊 Exploratory Time Series Analysis

The project examined the historical behavior of taxi orders through several visualizations.

### Demand Trends

The hourly demand plot showed substantial fluctuations and an overall upward trend during the observed period.

### Moving Average

A 24-hour rolling average was calculated to make broader demand patterns easier to identify.

### Time Series Decomposition

The series was decomposed into:

- **Trend:** Long-term changes in demand.
- **Seasonality:** Recurring patterns over time.
- **Residuals:** Variations not explained by trend or seasonality.

A seasonal period of 24 hours was used to examine daily patterns.

## ⚙️ Feature Engineering

Time-based features were created to help machine learning models capture historical demand patterns.

### Calendar Features

- Month
- Day of the month
- Day of the week
- Hour of the day

### Lag Features

Lagged order counts from the previous 24 hours were included.

These features allow the model to use past observations when estimating future demand.

### Rolling Mean

A 24-hour rolling mean was calculated using historical observations.

The rolling calculation applied `shift(1)` to prevent the target hour's actual demand from entering the predictor values.

**This is an important step for avoiding target leakage.**

## 🔀 Train-Test Split

The dataset was divided chronologically:

- **90%** for training.
- **10%** for testing.

The observations were not shuffled.

This preserves the temporal sequence and allows the models to be evaluated on later observations.

## 🤖 Machine Learning Models

Four regression models were evaluated.

### 1. Linear Regression

A baseline model used to evaluate whether a simple linear relationship could capture taxi demand patterns.

### 2. Decision Tree Regressor

A nonlinear model trained with different maximum depths.

### 3. Random Forest Regressor

An ensemble model combining multiple decision trees.

Different numbers of estimators were evaluated.

### 4. LightGBM Regressor

A gradient boosting model configured with:

- `n_estimators = 200`
- `learning_rate = 0.05`
- `max_depth = 10`
- `random_state = 12345`

## 📏 Evaluation Metric

Model performance was measured using **Root Mean Squared Error (RMSE)**:

\[
RMSE=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
\]

RMSE measures the magnitude of prediction errors in the same units as the target variable.

For this project, a lower RMSE indicates more accurate hourly taxi order predictions.

The company's performance requirement was:

**RMSE ≤ 48**

## 📈 Model Comparison Results

| Model | Reported Test RMSE | Meets Requirement? |
|---|---:|---|
| **LightGBM** | **40.53** | Yes |
| Random Forest | 43.70 | Yes |
| Linear Regression | 45.86 | Yes |
| Decision Tree | 50.47 | No |

### Key Findings

**LightGBM**

Achieved the lowest reported RMSE, approximately 7.47 orders below the required threshold.

**Random Forest**

Produced competitive predictions and met the performance requirement.

**Linear Regression**

Provided a useful baseline and also met the required RMSE.

**Decision Tree**

Produced the highest error among the evaluated models and did not meet the requirement.

## 🏆 Selected Model — LightGBM

LightGBM was selected as the best-performing model in the notebook.

Its reported performance was:

- **RMSE:** 40.53
- **Required maximum RMSE:** 48
- **Difference below the threshold:** Approximately 7.47 orders

The final analysis also included a comparison of actual and predicted values and a histogram of prediction errors.

These visualizations helped examine how well the model followed changes in demand.

## 💡 Business Applications

**Driver Allocation**

Forecasting demand can help estimate how many drivers may be needed during upcoming hours.

**Peak-Hour Planning**

Identifying recurring periods of higher demand can support operational planning.

**Service Efficiency**

More accurate forecasts may help reduce mismatches between driver availability and customer demand.

**Resource Optimization**

Demand forecasts can provide useful inputs for scheduling and staffing decisions.

These are potential business applications; operational improvements were not directly measured in the notebook.

## ⚠️ Project Limitations

- The analysis is based on historical taxi demand data.
- Forecast quality may change when demand patterns shift.
- Weather, holidays, and special events were not included as external predictors.
- The notebook uses test-set performance during model selection and hyperparameter comparison, which can lead to optimistic performance estimates.
- A separate chronological validation period or rolling time series cross-validation would strengthen the evaluation.
- The model was not deployed or evaluated in a live forecasting environment.

## 🚀 How to Run the Project

1. Clone or download this repository.
2. Install the required dependencies.
3. Place `taxi.csv` in the `data/` directory.
4. Update the dataset path in the notebook.
5. Open the notebook using Jupyter Notebook or JupyterLab.
6. Execute the cells in order.

### Install Dependencies

```bash
pip install pandas numpy matplotlib scikit-learn lightgbm statsmodels notebook
```

## 🔮 Future Improvements

- Implement `TimeSeriesSplit` for chronological cross-validation.
- Separate training, validation, and final testing periods.
- Experiment with additional lag and rolling-window features.
- Add weather, holidays, and special-event indicators.
- Compare LightGBM with other forecasting approaches.
- Develop an interactive dashboard to visualize predicted demand.
- Monitor forecasting accuracy over time.

## 📈 Conclusion

This project demonstrates how time series analysis and machine learning can be combined to forecast short-term transportation demand.

Historical taxi orders were aggregated into hourly intervals, analyzed for trends and seasonality, and transformed into features suitable for regression models.

Among the four evaluated models, **LightGBM achieved the lowest reported RMSE of 40.53**, meeting the project's performance requirement of 48.

The project showcases practical skills in **time series preprocessing, feature engineering, regression modeling, model evaluation, and business-oriented forecasting**.

## 👩‍💻 Author

**Andressa Miranda**

Computer Science | Data Analytics | Python | Machine Learning

[GitHub Profile](https://github.com/Andressa202)

*Developed as part of the TripleTen Data Science program.*
