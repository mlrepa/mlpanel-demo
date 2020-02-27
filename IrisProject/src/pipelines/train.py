import argparse
import joblib
import mlflow
from mlflow import log_artifact, log_param
import mlflow.sklearn
from mlflow.tracking.fluent import log_params
import os
from typing import Text
import yaml

from src.data.dataset import get_dataset
from src.train.train import train


def train_model(config_path: Text):

    config = yaml.load(open(config_path), Loader=yaml.FullLoader)

    estimator_name = config['train']['estimator_name']
    param_grid = config['train']['estimators'][estimator_name]['param_grid']
    cv = config['train']['cv']

    target_column = config['featurize']['target_column']
    train_df = get_dataset(config['split_train_test']['train_csv'])

    model = train(
        df=train_df,
        target_column=target_column,
        estimator_name=estimator_name,
        param_grid=param_grid,
        cv=cv
    )

    print(model.best_score_)

    model_name = config['base']['model']['model_name']
    models_folder = config['base']['model']['models_folder']

    joblib.dump(
        model,
        os.path.join(models_folder, model_name)
    )

    # Logging into mlflow

    with mlflow.start_run() as run:

        print(run)
        print(run.info)
        print(run.info.run_uuid)

        param_grid = config['train']['estimators'][estimator_name]['param_grid']

        log_param(key='estimator', value=estimator_name)
        log_param(key='cv', value=config['train']['cv'])
        log_params(param_grid)

        mlflow.sklearn.log_model(model, 'model')

        log_artifact(local_path='src/features/features.py')
        log_artifact(local_path='src/train/train.py')
        log_artifact(local_path='src/pipelines/train.py')


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--train-config', dest='config', required=True)
    args = args_parser.parse_args()

    train_model(config_path=args.config)

