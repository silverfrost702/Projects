SmogCast: Air Quality Prediction using XGBoost, LSTM & Transformers

SmogCast is an end-to-end forecasting and classification pipeline for Air Quality Index (AQI) using classical ML and modern deep learning approaches.
This project predicts air pollutant concentrations (PM2.5 & PM10) for the next 7 days and classifies AQI levels using XGBoost.

Models Implemented

XGBoost Classifier – Predicts AQI categories

LSTM Forecaster – Multi-step, multi-feature pollutant forecasting

Transformer Encoder Forecaster – Attention-based time series prediction

Persistence Baseline – Benchmark for comparison

SHAP Explainability – Feature importance for XGBoost

Dataset

Source: city_hour.csv

Preprocessing steps:

City-wise filtering (Delhi)

Missing value handling

Daily aggregation (mean)

Lag features (1, 3, 7, 14 days)

Rolling windows (3, 7, 14 days)

Calendar features (sin/cos encoding)

Pipeline Overview

Preprocess hourly raw data → daily pollutant features

Engineer time-series features

Train XGBoost AQI classifier

Prepare multi-step sequences

Train & evaluate LSTM

Train & evaluate Transformer

Extract attention heatmaps

Compute SHAP explainability

Save trained models

Model Comparison
Model	MAE	RMSE	Notes
Persistence Baseline	Highest	Highest	Benchmark
LSTM	Lower than baseline	Lower	Learns temporal structure
Transformer Encoder	Lowest	Lowest	Best multi-step forecasting
XGBoost (Classification)	High accuracy	–	Best for AQI categories

(Fill with your actual numbers after training)

How to Run
python smogcast_final_fixed.py

Outputs

models/lstm_model.keras

models/transformer_model.keras

Attention heatmaps

SHAP summary plots

Forecast evaluation charts

Project Structure
├── smogcast_final_fixed.py
├── city_hour.csv
├── models/
│   ├── lstm_model.keras
│   └── transformer_model.keras
└── README.md
