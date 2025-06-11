
import pandas as pd
import numpy as np
import time
import shutil
import os
from flaml import AutoML
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

dosyalar = [f"Cell{i}_Cycle_Peak_meanT_SoH.csv" for i in range(1, 9)]
results = []

for i in range(len(dosyalar)):
    print(f"\n Test dosyası: {dosyalar[i]}")

    test_df = pd.read_csv(dosyalar[i])
    train_dfs = [pd.read_csv(d) for j, d in enumerate(dosyalar) if j != i]
    train_df = pd.concat(train_dfs, ignore_index=True)

    X_train = train_df[["Cycle", "Peak", "Mean_Temperature"]]
    y_train = train_df["SoH"]
    X_test = test_df[["Cycle", "Peak", "Mean_Temperature"]]
    y_test = test_df["SoH"]

    automl = AutoML()
    settings = {
        "time_budget": 900,  # saniye cinsinden
        "metric": "rmse",
        "task": "regression",
        "log_file_name": f"flaml_log_cell{i}.log",
        "verbose": 0,
        "n_splits": 10
    }

    start_train = time.time()
    automl.fit(X_train=X_train, y_train=y_train, **settings)
    end_train = time.time()
    train_time = end_train - start_train

    start_pred = time.time()
    y_pred = automl.predict(X_test)
    end_pred = time.time()
    predict_time = end_pred - start_pred

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    best_model = str(automl.model.estimator)

    results.append({
        "Test_Cell": dosyalar[i],
        "Time_Budget_sec": 900,
        "Train_Time_sec": round(train_time, 2),
        "Predict_Time_sec": round(predict_time, 2),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R2": round(r2, 4),
        "MAPE": round(mape, 4),
        "Best_Model": best_model,
        "Best_HPO_Algorithm": "FLAML (10-fold CV, kaynak dostu)"
    })

df_results = pd.DataFrame(results)
print("\n Tüm sonuçlar:")
print(df_results)
df_results.to_csv("900_flaml_soh_results.csv", index=False)
