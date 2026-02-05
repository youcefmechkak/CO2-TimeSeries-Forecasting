# PrediCO2 – CO₂ Forecasting Using Machine Learning and Deep Learning

## Project Overview
This project focuses on forecasting and analyzing CO₂ concentration levels using classical time-series models and deep learning approaches. The objective is to compare multiple models in terms of accuracy, computational cost, and complexity, following the main steps of the data mining pipeline.

## Models Implemented
- ARIMA
- SARIMA
- LSTM
- Transformer for time series

## Dataset
The dataset contains historical CO₂ concentration measurements over time.
It is located in the `data/` directory.

## Methodology
1. **Data Exploration**
   - Time-series visualization
   - Distribution analysis
   - Correlation analysis
   - Missing value inspection

2. **Preprocessing**
   - Missing value handling
   - Feature scaling
   - Time-series windowing

3. **Validation Strategy**
   - Rolling window validation for time series
   - Train / validation / test split

4. **Evaluation Metrics**
   - MSE
   - RMSE
   - MAE
   - R²
   - Training time
   - Inference time
   - Number of trainable parameters

5. **Forecasting**
   - One-year CO₂ level prediction
   - Visual comparison across models

## Repository Structure
- `src/`: Python implementation (`PrediCO2.py`)
- `notebook/`: Jupyter notebook for experimentation
- `data/`: Dataset
- `results/`: Metrics and plots
- `report/`: Project report (PDF)
- `video/`: Model explanation video

## Requirements
Install dependencies using:
```bash
pip install -r requirements.txt
