# TSAlchemy

TSAlchemy is a comprehensive time series analysis toolkit that provides various tools and utilities for time series data manipulation, analysis, and modeling.

## Features

- Time series data preprocessing and manipulation
- Seasonality detection and decomposition
- Cycle phase analysis
- Dynamic Factor Models (DFM)
- Impulse Response Function (IRF) analysis
- Various time series metrics and statistics

## Installation

```bash
pip install .
```

## Usage

```python
import tsalchemy as ts

# Time series tools
ts.check_time_series(df)
ts.process_raw_yoy(series)
ts.determine_cycle_phase(series)

# Dynamic Factor Model
model = ts.SingleFactorDFM(df_data)
model.build_model()

# Impulse Response Function
irf = ts.IRF(data, lags=2)
irf.compute_irf()
```

## Dependencies

- pandas
- numpy
- matplotlib
- scikit-learn
- statsmodels
- scipy
- fastdtw

## Version

0.1.0 