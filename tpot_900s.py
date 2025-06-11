import pandas as pd
import numpy as np
import time
from tpot import TPOTRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


dosyalar = [f"Cell{i}_Cycle_Peak_meanT_SoH.csv" for i in range(1, 9)]


results = []

for i in range(len(dosyalar)):
    print(f"\n Test dosyası: {dosyalar[i]}")
    
    # Eğitim ve test verisini ayır
    test_df = pd.read_csv(dosyalar[i])
    train_dfs = [pd.read_csv(d) for j, d in enumerate(dosyalar) if j != i]
    train_df = pd.concat(train_dfs, ignore_index=True)

    
    X_train = train_df[["Cycle", "Peak", "Mean_Temperature"]]
    y_train = train_df["SoH"]
    X_test = test_df[["Cycle", "Peak", "Mean_Temperature"]]
    y_test = test_df["SoH"]

    
    time_budget = 900
    tpot = TPOTRegressor(
        generations=100,
        population_size=50,
        verbosity=2,
        random_state=42,
        cv=10,
        max_time_mins=time_budget / 60,
        n_jobs=-1,
        scoring='neg_mean_squared_error',
        warm_start=True
    )


    start_train = time.time()
    tpot.fit(X_train, y_train)
    end_train = time.time()
    train_time = end_train - start_train

    start_pred = time.time()
    y_pred = tpot.predict(X_test)
    end_pred = time.time()
    predict_time = end_pred - start_pred

 
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    best_model = str(tpot.fitted_pipeline_.steps[-1][1].__class__.__name__)

    results.append({
        "Test_Cell": dosyalar[i],
        "Time_Budget_sec": time_budget,
        "Train_Time_sec": round(train_time, 2),
        "Predict_Time_sec": round(predict_time, 2),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R2": round(r2, 4),
        "MAPE": round(mape, 4),
        "Best_Model": best_model,
        "Best_HPO_Algorithm": "Genetic Programming"
    })


df_results = pd.DataFrame(results)
print("\n Tüm sonuçlar:")
print(df_results)

df_results.to_csv("tpot_soh_results.csv", index=False)
