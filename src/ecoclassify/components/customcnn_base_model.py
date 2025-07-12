import os
import torch
import torch.nn as nn
import mlflow
import dagshub
from ecoclassify.config.configuration import BaseModelConfig
from ecoclassify import logger
from torchinfo import summary


class CustomCNN(nn.Module):
    def __init__(self, dropout, hidden_units, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64 * 28 * 28, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class PrepareCustomCNN:
    def __init__(self, config: BaseModelConfig):
        self.config = config

    def run(self):
        model = CustomCNN(
            dropout=self.config.dropout,
            hidden_units=self.config.hidden_units,
            num_classes=self.config.num_classes
        )

        # Create model dir
        os.makedirs(self.config.root_dir, exist_ok=True)
        torch.save(model.state_dict(), self.config.model_path)

        # MLflow tracking
        dagshub.init(self.config.dagshub_repo_owner, self.config.dagshub_repo_name, mlflow=True)
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        mlflow.set_experiment(self.config.mlflow_experiment_name)

        with mlflow.start_run(run_name="customcnn_baseline"):
            mlflow.log_params({
                "model_name": self.config.model_name,
                "dropout": self.config.dropout,
                "hidden_units": self.config.hidden_units,
                "num_classes": self.config.num_classes
            })

            mlflow.log_artifact(self.config.model_path)
            model_summary = str(summary(model, input_size=(1, 3, 224, 224)))
            with open(os.path.join(self.config.root_dir, "summary.txt"), "w") as f:
                f.write(model_summary)
            mlflow.log_artifact(os.path.join(self.config.root_dir, "summary.txt"))

        logger.info(f" CustomCNN baseline saved and logged.")
