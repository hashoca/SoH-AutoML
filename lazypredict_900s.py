import pandas as pd
import numpy as np
import time
from lazypredict.Supervised import LazyRegressor
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


    reg = LazyRegressor(verbose=0, ignore_warnings=True, predictions=True)

    start_train = time.time()
    models, predictions = reg.fit(X_train, X_test, y_train, y_test)
    end_train = time.time()
    train_time = end_train - start_train


    best_model_name = models.sort_values("RMSE").index[0]
    y_pred = predictions[best_model_name].values


    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    results.append({
        "Test_Cell": dosyalar[i],
        "Time_Budget_sec": 900,
        "Train_Time_sec": round(train_time, 2),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R2": round(r2, 4),
        "MAPE": round(mape, 4),
        "Best_Model": best_model_name,
        "Best_HPO_Algorithm": "LazyPredict (Top model selection, no tuning)"
    })

df_results = pd.DataFrame(results)
print("\n Tüm sonuçlar:")
print(df_results)
df_results.to_csv("900_lazypredict_soh_results.csv", index=False)
