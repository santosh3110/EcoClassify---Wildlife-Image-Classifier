import torch
import torch.nn as nn
import torchvision
import mlflow
import dagshub
from ecoclassify.entity.config_entity import ResNetModelConfig, TrainingConfig
from ecoclassify import logger


class ResNet50Model:
    def __init__(self, config: ResNetModelConfig, train_config: TrainingConfig):
        self.config = config
        self.train_config = train_config

    def get_model(self):
        logger.info("⏬ Loading pretrained ResNet50...")
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)

        for name, param in model.named_parameters():
            if "layer4" in name or "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        model.fc = nn.Linear(model.fc.in_features, 8)  # 8 species
        return model

    def save_model(self, model: torch.nn.Module):
        torch.save(model.state_dict(), self.config.model_path)
        logger.info(f"💾 ResNet50 base model saved at: {self.config.model_path}")

    def log_model_mlflow(self, model: torch.nn.Module):
        try:
            dagshub.init(self.config.dagshub_repo_owner, self.config.dagshub_repo_name, mlflow=True)
        except Exception as e:
            print(f"⚠️ DAGsHub init failed (likely already exists): {e}")
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        mlflow.set_experiment(self.config.mlflow_experiment_name)

        with mlflow.start_run():
            mlflow.log_param("model", "ResNet50")
            mlflow.pytorch.log_model(model, artifact_path="resnet50_model")
            logger.info("✅ ResNet50 logged to MLflow via DAGsHub.")
