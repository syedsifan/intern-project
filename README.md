# XAU Analytics

A hosted local app for analyzing and predicting gold (XAU) prices using machine learning and historical market data.

## Project Overview

XAU Analytics leverages historical gold price data and related financial features to train a regression model that estimates future gold prices. The app provides an interactive web interface for price predictions and analysis.

## Features

- Load gold price dataset from `gld_price_data.csv`
- Train a Random Forest regression model
- Display model performance metrics (R², MAE, RMSE)
- Interactive prediction form with real-time input
- Feature statistics and analysis

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- XGBoost

## Getting Started

### Prerequisites

- Python 3.8 or higher

### Installation

Install the required Python libraries:

```powershell
python -m pip install -r requirements.txt
```

### Run the App

Start the hosted app:

```powershell
streamlit run app.py
```

Then open the local URL shown in the terminal, typically `http://localhost:8501`.

## Notes

- The web app trains the model at startup using the local dataset.
- This is a local host deployment for demonstration and testing.
