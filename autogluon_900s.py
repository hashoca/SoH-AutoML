import pandas as pd
import numpy as np
import time
from autogluon.tabular import TabularPredictor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import os
import shutil


dosyalar = [f"Cell{i}_Cycle_Peak_meanT_SoH.csv" for i in range(1, 9)]
results = []

for i in range(len(dosyalar)):
    print(f"\n Test dosyası: {dosyalar[i]}")


    test_df = pd.read_csv(dosyalar[i])
    train_dfs = [pd.read_csv(d) for j, d in enumerate(dosyalar) if j != i]
    train_df = pd.concat(train_dfs, ignore_index=True)


    train_data = train_df.copy()
    test_data = test_df.copy()


    save_path = f"autogluon_temp_fold_{i}"
    if os.path.exists(save_path):
        shutil.rmtree(save_path)

    time_limit = 900  # saniye
    predictor = TabularPredictor(
        label="SoH",
        problem_type="regression",
        path=save_path,
        eval_metric="root_mean_squared_error"
    )

    start_train = time.time()
    predictor.fit(
        train_data,
        presets='best_quality',
        time_limit=time_limit,
        num_bag_folds=10,  # 10-fold CV
        num_bag_sets=1,    # Bagging tekrar sayısı
        num_stack_levels=1, # Stack seviyesi
        verbosity=2
    )
    end_train = time.time()
    train_time = end_train - start_train

    X_test = test_data[["Cycle", "Peak", "Mean_Temperature"]]
    start_pred = time.time()
    y_pred = predictor.predict(X_test).values
    end_pred = time.time()
    predict_time = end_pred - start_pred

    y_test = test_data["SoH"].values

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    leaderboard = predictor.leaderboard(test_data, silent=True)
    best_model = leaderboard.iloc[0]["model"]

    results.append({
        "Test_Cell": dosyalar[i],
        "Time_Budget_sec": time_limit,
        "Train_Time_sec": round(train_time, 2),
        "Predict_Time_sec": round(predict_time, 2),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R2": round(r2, 4),
        "MAPE": round(mape, 4),
        "Best_Model": best_model,
        "Best_HPO_Algorithm": "AutoGluon (Bagging+Stacking, GPU aktif)"
    })

    shutil.rmtree(save_path)


df_results = pd.DataFrame(results)
print("\n Tüm sonuçlar:")
print(df_results)
df_results.to_csv("900_autogluon_soh_results.csv", index=False)