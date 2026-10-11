# ZTM backend - extended benchmark (builds on the baseline repo)
Run (CMD):  pip install numpy pandas scikit-learn lightgbm catboost
            python -m src.download_data
            python -m src.task1_traffic_forecasting     (baseline)
            python -m src.task2_anomaly_detection       (baseline)
            python -m src.task3_resource_allocation_rl  (baseline)
            python -m src.benchmark --folds 5 --lstm-epochs 12   (improved, recommended)
Outputs in outputs\ (json + csv).
