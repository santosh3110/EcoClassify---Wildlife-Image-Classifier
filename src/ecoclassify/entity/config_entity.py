from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass
class BaseModelConfig:
    root_dir: Path
    model_name: str
    model_path: Path
    num_classes: int
    dropout: float
    hidden_units: int
    mlflow_tracking_uri: str
    mlflow_experiment_name: str
    dagshub_repo_owner: str
    dagshub_repo_name: str


@dataclass
class ResNetModelConfig:
    root_dir: Path
    model_name: str
    model_path: Path
    mlflow_tracking_uri: str
    mlflow_experiment_name: str
    dagshub_repo_owner: str
    dagshub_repo_name: str


@dataclass
class PrepareCallbacksConfig:
    root_dir: Path
    tensorboard_root_log_dir: Path
    checkpoint_model_filepath: Path


@dataclass
class TrainingConfig:
    root_dir: Path
    customcnn_trained_model_path: Path
    resnet_trained_model_path: Path
    log_dir: Path
    epochs: int
    lr: float
    weight_decay: float
    alpha_l2sp: float
    batch_size: int
    image_size: List[int]
    label_smoothing: float
    grad_clip: float
    early_stopping_patience: int
    scheduler_patience: int
    scheduler_factor: float
    model_to_use: str


@dataclass
class EvaluationConfig:
    root_dir: Path
    report_path: Path


@dataclass
class TemperatureTuningConfig:
    enabled: bool
    search_range: List[float]
    search_steps: int
