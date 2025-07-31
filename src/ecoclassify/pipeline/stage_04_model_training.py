from ecoclassify.config.configuration import ConfigurationManager
from ecoclassify.components.training import ModelTrainer
from ecoclassify.components.data_loader import get_dataloaders
from ecoclassify.components.customcnn_base_model import CustomCNN
from ecoclassify.components.resnet50_model import ResNet50Model
from ecoclassify.utils.logger import get_logger
import pandas as pd
import torch
import os

STAGE_NAME = "Model Training Stage"
logger = get_logger(STAGE_NAME.replace(" ", "_").lower())


class ModelTrainingPipeline:
    def __init__(self):
        self.config = ConfigurationManager()
        self.data_ingestion_config = self.config.get_data_ingestion_config()
        self.train_config = self.config.get_training_config()
        self.prepare_callbacks_config = self.config.get_prepare_callbacks_config()

    def main(self):
        df_path = os.path.join(self.data_ingestion_config.unzip_dir, "train_split.csv")
        labels_path = os.path.join(self.data_ingestion_config.unzip_dir, "train_labels.csv")

        df = pd.read_csv(df_path, index_col="id")
        labels_df = pd.read_csv(labels_path, index_col="id")

        # Add label column
        df["label"] = labels_df.loc[df.index].values.argmax(1)

        # Split train & val
        train_df = df[df["split"] == "train"]
        val_df = df[df["split"] == "val"]

        # Dataloaders
        train_loader, val_loader = get_dataloaders(
            train_df=train_df,
            val_df=val_df,
            batch_size=self.train_config.batch_size,
            config=self.train_config
        )

        # Model selection
        if self.train_config.model_to_use.lower() == "customcnn":
            model = CustomCNN(
                dropout=self.train_config.dropout,
                hidden_units=self.train_config.hidden_units,
                num_classes=self.train_config.num_classes
            )
            model_path = self.train_config.customcnn_trained_model_path
            experiment = "prepare_customcnn"
        else:
            model = ResNet50Model(
                config=self.config.get_resnet_model_config(),
                train_config=self.train_config
            ).get_model()
            model_path = self.train_config.resnet_trained_model_path
            experiment = "train_resnet50"

        trainer = ModelTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=self.train_config,
            model_path=model_path,
            experiment_name=experiment,
            callbacks_config=self.prepare_callbacks_config
        )

        trainer.train()


if __name__ == "__main__":
    try:
        logger.info(f"🔁 STARTING {STAGE_NAME}")
        pipeline = ModelTrainingPipeline()
        pipeline.main()
        logger.info(f"✅ COMPLETED {STAGE_NAME}")
    except Exception as e:
        logger.exception(f"{STAGE_NAME} FAILED due to: {e}")
        raise e
