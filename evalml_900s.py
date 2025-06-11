import os
import shutil
import time
import numpy as np
import pandas as pd
from pycaret.regression import setup, compare_models, predict_model, pull
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

dosyalar = [f"Cell{i}_Cycle_Peak_meanT_SoH.csv" for i in range(1, 9)]
results = []

for i in range(len(dosyalar)):
    print(f"\n Test dosyası: {dosyalar[i]}")

    test_df = pd.read_csv(dosyalar[i])
    train_dfs = [pd.read_csv(d) for j, d in enumerate(dosyalar) if j != i]
    train_df = pd.concat(train_dfs, ignore_index=True)

    exp_folder = f"pycaret_exp_{i}"
    if os.path.exists(exp_folder):
        shutil.rmtree(exp_folder)
    os.makedirs(exp_folder, exist_ok=True)
    os.chdir(exp_folder)

    exp = setup(
        data=train_df,
        target="SoH",
        train_size=1.0,
        fold_strategy="kfold",
        fold=10,
        session_id=42,
        use_gpu=True,
        n_jobs=-1,
        silent=True,
        verbose=False
    )

    start_train = time.time()
    best_model = compare_models(n_select=1, sort="RMSE", turbo=True)
    end_train = time.time()
    train_time = end_train - start_train

    X_test = test_df[["Cycle", "Peak", "Mean_Temperature"]].copy()
    y_test = test_df["SoH"].values

    start_pred = time.time()
    preds_df = predict_model(best_model, data=X_test)
    end_pred = time.time()
    predict_time = end_pred - start_pred
    y_pred = preds_df["Label"].values

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    model_info = pull()
    model_name = model_info.loc[0, 'Model']

    results.append({
        "Test_Cell": dosyalar[i],
        "Time_Budget_sec": 900,
        "Train_Time_sec": round(train_time, 2),
        "Predict_Time_sec": round(predict_time, 2),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R2": round(r2, 4),
        "MAPE": round(mape, 4),
        "Best_Model": model_name,
        "Best_HPO_Algorithm": "PyCaret (CV + Ensemble, GPU varsa aktif)"
    })

    os.chdir("..")
    shutil.rmtree(exp_folder)

df_results = pd.DataFrame(results)
print("\n Tüm sonuçlar:")
print(df_results)
df_results.to_csv("pycaret_soh_results.csv", index=False)
