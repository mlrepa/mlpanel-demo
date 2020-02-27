import argparse
import joblib
import json
import mlflow
from mlflow import log_metric
import os
from typing import Text
import yaml

from src.data.dataset import get_dataset
from src.evaluate.evaluate import evaluate


def evaluate_model(config_path: Text):

    config = yaml.load(open(config_path), Loader=yaml.FullLoader)

    target_column = config['featurize']['target_column']
    test_df = get_dataset(config['split_train_test']['test_csv'])
    model_name = config['base']['model']['model_name']
    models_folder = config['base']['model']['models_folder']

    model = joblib.load(os.path.join(models_folder, model_name))

    f1, cm = evaluate(df=test_df,
                      target_column=target_column,
                      clf=model)

    test_report = {
        'f1_score': f1,
        'confusion_matrix': cm.tolist()
    }
    print(test_report)
    filepath = os.path.join(config['base']['experiments']['experiments_folder'],
                            config['evaluate']['metrics_file'])
    json.dump(obj=test_report, fp=open(filepath, 'w'), indent=2)

    # Logging into mlflow
    with mlflow.start_run() as run:

        print(run)
        print(run.info)
        print(run.info.run_uuid)

        log_metric(key='f1_score', value=f1)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--evaluate-config', dest='config', required=True)
    args = args_parser.parse_args()

    evaluate_model(config_path=args.config)
