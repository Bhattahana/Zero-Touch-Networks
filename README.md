# Zero-Touch Network Management (ZTM) — Extended Benchmark

An ML-driven Zero-Touch Network Management system for 5G networks, integrating traffic forecasting, anomaly detection, and reinforcement learning-based resource allocation. This project extends the baseline implementation with an experimental benchmarking framework to evaluate and compare different models and network optimization strategies.

## Features

- **Traffic Forecasting:** Predicts future network traffic using machine learning and time-series forecasting models.
- **Anomaly Detection:** Identifies unusual network traffic patterns and potential network anomalies.
- **Resource Allocation:** Uses Q-learning to optimize resource allocation and compares its performance against baseline allocation strategies.
- **Extended Benchmarking:** Evaluates model performance using cross-validation and configurable LSTM training epochs.
- **Performance Evaluation:** Generates JSON and CSV reports for analyzing and comparing experimental results.

## Installation

Install the required Python dependencies:

```bash
pip install numpy pandas scikit-learn lightgbm catboost
```

Ensure that Python and pip are installed and available in your system PATH.

## Running the Project

Run the following commands from the project root directory.

### 1. Download the Dataset

```bash
python -m src.download_data
```

### 2. Run Baseline Experiments

**Task 1: Traffic Forecasting**
```bash
python -m src.task1_traffic_forecasting
```

**Task 2: Anomaly Detection**
```bash
python -m src.task2_anomaly_detection
```

**Task 3: Reinforcement Learning-Based Resource Allocation**
```bash
python -m src.task3_resource_allocation_rl
```

### 3. Run the Extended Benchmark

The extended benchmark evaluates the experimental pipeline using five folds and 12 LSTM training epochs.

```bash
python -m src.benchmark --folds 5 --lstm-epochs 12
```

Adjust the number of folds and LSTM epochs as required.

## Output

Experimental outputs are saved in the `outputs/` directory in JSON and CSV formats. These files can be used to analyze model performance, compare resource allocation strategies, and review benchmarking results.

## Project Structure

```text
.
├── src/
│   ├── download_data.py
│   ├── task1_traffic_forecasting.py
│   ├── task2_anomaly_detection.py
│   ├── task3_resource_allocation_rl.py
│   └── benchmark.py
├── outputs/
└── README.md
```

*Note: The structure above illustrates the main entry points. Additional files and directories may exist in the repository.*

## Project Status

The project includes baseline implementations for traffic forecasting, anomaly detection, and reinforcement learning-based resource allocation, alongside an extended benchmarking pipeline.

Further work can focus on improving model accuracy, expanding baseline comparisons, integrating live network telemetry, and exploring deployment with network controllers.

## Technologies

- Python
- NumPy and Pandas
- Scikit-learn
- LightGBM
- CatBoost
- LSTM-based traffic forecasting
- Q-learning for resource allocation

## Future Work

- Improve forecasting accuracy and anomaly detection performance.
- Evaluate additional machine learning and reinforcement learning algorithms.
- Expand benchmarking across datasets and network conditions.
- Integrate real-time telemetry and automated network management capabilities.
- Explore deployment with programmable network controllers.

## Disclaimer

This project is an experimental research implementation. Real-time network integration and fully autonomous production deployment should not be assumed unless separately implemented and validated.
