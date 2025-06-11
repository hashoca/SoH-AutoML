import pandas as pd
import numpy as np
import time
import h2o
from h2o.automl import H2OAutoML
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

h2o.init(port=54325, max_mem_size="2G")


dosyalar = [f"Cell{i}_Cycle_Peak_meanT_SoH.csv" for i in range(1, 9)]
results = []


for i in range(len(dosyalar)):
    print(f"\n Test dosyası: {dosyalar[i]}")


    test_df = pd.read_csv(dosyalar[i])
    train_dfs = [pd.read_csv(d) for j, d in enumerate(dosyalar) if j != i]
    train_df = pd.concat(train_dfs, ignore_index=True)


    train_h2o = h2o.H2OFrame(train_df)
    test_h2o = h2o.H2OFrame(test_df)

    features = ["Cycle", "Peak", "Mean_Temperature"]
    target = "SoH"


    aml = H2OAutoML(
        max_runtime_secs=900,
        nfolds=10,
        sort_metric="RMSE",
        seed=42,
        exclude_algos=["XGBoost"],  # ⛔ Java 19 ile uyumsuz olan algoritma dışlandı
        verbosity="warn"
    )

    start_train = time.time()
    aml.train(x=features, y=target, training_frame=train_h2o)
    end_train = time.time()
    train_time = end_train - start_train


    start_pred = time.time()
    preds = aml.predict(test_h2o).as_data_frame().values.flatten()
    end_pred = time.time()
    predict_time = end_pred - start_pred


    y_test = test_df[target].values
    y_pred = preds
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)


    leader_model_id = aml.leader.model_id
    if "GBM" in leader_model_id:
        model_type = "GBM"
    elif "DRF" in leader_model_id:
        model_type = "Random Forest"
    elif "StackedEnsemble" in leader_model_id:
        model_type = "StackedEnsemble"
    elif "DeepLearning" in leader_model_id:
        model_type = "Deep Learning"
    elif "GLM" in leader_model_id:
        model_type = "GLM"
    else:
        model_type = leader_model_id

    results.append({
        "Test_Cell": dosyalar[i],
        "Time_Budget_sec": 900,
        "Train_Time_sec": round(train_time, 2),
        "Predict_Time_sec": round(predict_time, 2),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R2": round(r2, 4),
        "MAPE": round(mape, 4),
        "Best_Model": model_type,
        "Best_HPO_Algorithm": "H2O AutoML (XGBoost hariç)"
    })

df_results = pd.DataFrame(results)
print("\n Tüm sonuçlar:")
print(df_results)

df_results.to_csv("3600_h2o_soh_results_no_xgboost.csv", index=False)