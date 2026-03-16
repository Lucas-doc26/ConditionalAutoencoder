import mlflow
import pandas as pd
from src.config import *

Config = Config()

dataframe_final = pd.DataFrame()

for type in ["AE", "Skip"]:
    EXPERIMENT_NAME = f"Classifier_{type}_0"
    OUTPUT_CSV = "metrics_classifiers_1.csv"

    mlflow.set_tracking_uri(Config.IP_LOCAL)

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        raise ValueError(f"Experimento '{EXPERIMENT_NAME}' não encontrado.")

    experiment_id = experiment.experiment_id

    df = mlflow.search_runs(
        experiment_ids=[experiment_id],
        output_format="pandas"
    )

    dataframe_final = pd.concat([dataframe_final, df], ignore_index=True)

print(f"✅ {len(dataframe_final)} runs exportados para {OUTPUT_CSV}")
dataframe_final.to_csv(OUTPUT_CSV, index=False)