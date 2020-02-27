
import mlflow
import yaml


if __name__ == '__main__':

    config = yaml.load(open('config/pipeline_config.yml'), Loader=yaml.FullLoader)
    estimator_name = config['train']['estimator_name']
    mlflow.set_experiment(estimator_name)

    client = mlflow.tracking.MlflowClient()
    experiment_id = client.get_experiment_by_name(estimator_name).experiment_id
    mlflow.run(uri='.', experiment_id=experiment_id, use_conda=False)
