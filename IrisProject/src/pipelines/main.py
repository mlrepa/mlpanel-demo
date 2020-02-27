import argparse
import mlflow
from typing import Text, Dict


def get_run(entrypoint: Text, parameters: Dict, run_uuid: Text) -> None:
    """Run an entrypoint with specified run_uuid"""

    submitted_run = mlflow.run(".", entrypoint, parameters=parameters, run_id=run_uuid, use_conda=False)
    mlflow.tracking.MlflowClient().get_run(submitted_run.run_id)


def workflow(config_path: Text) -> None:

    with mlflow.start_run() as active_run:

        run_uuid = active_run.info.run_uuid

        get_run("featurize", {'featurize-config': config_path}, run_uuid=run_uuid)
        get_run("split_train_test", {'split-train-test-config': config_path}, run_uuid=run_uuid)
        get_run("train", {'train-config': config_path}, run_uuid=run_uuid)
        get_run("evaluate", {'evaluate-config': config_path}, run_uuid=run_uuid)


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    workflow(config_path=args.config)