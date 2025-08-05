from ecoclassify.utils.common import read_yaml, create_directories
from ecoclassify.constants.paths import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from ecoclassify.entity.config_entity import (
    DataIngestionConfig,
    BaseModelConfig,
    ResNetModelConfig,
    PrepareCallbacksConfig,
    TrainingConfig,
    EvaluationConfig,
    TemperatureTuningConfig,
)
from pathlib import Path


class ConfigurationManager:
    def __init__(self, 
                 config_filepath: Path = CONFIG_FILE_PATH, 
                 params_filepath: Path = PARAMS_FILE_PATH):
        
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([Path(self.config.artifacts_root)])

    def get_mlflow_config(self, stage: str) -> dict:
        return {
            "tracking_uri": self.config.mlflow.tracking_uri,
            "experiment_name": self.config.experiments[stage]
        }

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([Path(config.root_dir)])

        return DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=config.source_URL,
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir)
        )
    
    def get_base_model_config(self) -> BaseModelConfig:
        config = self.config.base_model
        model_params = self.params.CUSTOMCNN
        mlflow_cfg = self.config.mlflow
        return BaseModelConfig(
            root_dir=Path(config.root_dir),
            model_name=config.model_name,
            model_path=Path(config.model_path),
            num_classes=model_params.num_classes,
            dropout=model_params.dropout,
            hidden_units=model_params.hidden_units,
            mlflow_tracking_uri=mlflow_cfg.tracking_uri,
            mlflow_experiment_name=self.config.experiments.prepare_customcnn,
            dagshub_repo_owner=mlflow_cfg.repo_owner,
            dagshub_repo_name=mlflow_cfg.repo_name,
        )

    def get_resnet_model_config(self) -> ResNetModelConfig:
        config = self.config.resnet50_model
        mlflow_cfg = self.config.mlflow
        create_directories([Path(config.root_dir)])

        return ResNetModelConfig(
            root_dir=Path(config.root_dir),
            model_name=config.model_name,
            model_path=Path(config.model_path),
            mlflow_tracking_uri=mlflow_cfg.tracking_uri,
            mlflow_experiment_name=self.config.experiments.train_resnet50,
            dagshub_repo_owner=mlflow_cfg.repo_owner,
            dagshub_repo_name=mlflow_cfg.repo_name
        )

    def get_prepare_callbacks_config(self) -> PrepareCallbacksConfig:
        config = self.config.prepare_callbacks
        model_to_use = self.params.MODEL_TO_USE.lower()

        checkpoint_path = (
        config.customcnn_checkpoint_path if model_to_use == "customcnn"
        else config.resnet50_checkpoint_path
    )
        create_directories([
            Path(config.root_dir),
            Path(Path(checkpoint_path).parent)
        ])

        return PrepareCallbacksConfig(
            root_dir=Path(config.root_dir),
            checkpoint_model_filepath=Path(checkpoint_path)
        )

    def get_training_config(self) -> TrainingConfig:
        config = self.config.model_training
        params = self.params
        mlflow_cfg = self.config.mlflow

        create_directories([Path(config.root_dir), Path(config.log_dir)])

        return TrainingConfig(
            root_dir=Path(config.root_dir),
            customcnn_trained_model_path=Path(config.customcnn_trained_model_path),
            resnet_trained_model_path=Path(config.resnet_trained_model_path),
            log_dir=Path(config.log_dir),
            epochs=params.EPOCHS,
            lr=params.LEARNING_RATE,
            weight_decay=params.WEIGHT_DECAY,
            alpha_l2sp=params.ALPHA_L2SP,
            batch_size=params.BATCH_SIZE,
            image_size=params.IMAGE_SIZE,
            label_smoothing=params.LABEL_SMOOTHING,
            grad_clip=params.GRAD_CLIP,
            early_stopping_patience=params.EARLY_STOPPING_PATIENCE,
            scheduler_patience=params.SCHEDULER_PATIENCE,
            scheduler_factor=params.SCHEDULER_FACTOR,
            model_to_use=params.MODEL_TO_USE,
            dropout=params.CUSTOMCNN.dropout,
            hidden_units=params.CUSTOMCNN.hidden_units,
            num_classes=params.CUSTOMCNN.num_classes,
            mlflow_tracking_uri=mlflow_cfg.tracking_uri,
            dagshub_repo_owner=mlflow_cfg.repo_owner,
            dagshub_repo_name=mlflow_cfg.repo_name
        )

    def get_evaluation_config(self) -> EvaluationConfig:
        config = self.config.model_evaluation
        create_directories([Path(config.root_dir)])

        return EvaluationConfig(
            root_dir=Path(config.root_dir),
            report_path=Path(config.report_path)
        )

    def get_temperature_tuning_config(self) -> TemperatureTuningConfig:
        params = self.params.TEMPERATURE_TUNING

        return TemperatureTuningConfig(
            enabled=params.ENABLED,
            search_range=params.SEARCH_RANGE,
            search_steps=params.SEARCH_STEPS
        )
