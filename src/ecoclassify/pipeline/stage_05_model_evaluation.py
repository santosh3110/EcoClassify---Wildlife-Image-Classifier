import os
import pandas as pd
import torch

from ecoclassify.config.configuration import ConfigurationManager
from ecoclassify.components.data_loader import get_dataloaders
from ecoclassify.components.evaluation import ModelEvaluator
from ecoclassify.components.customcnn_base_model import CustomCNN
from ecoclassify.components.resnet50_model import ResNet50Model
from ecoclassify.utils.logger import get_logger

STAGE_NAME = "Model Evaluation"
logger = get_logger(STAGE_NAME.replace(" ", "_").lower())


class ModelEvaluationPipeline:
    def __init__(self):
        self.config = ConfigurationManager()
        self.eval_config = self.config.get_evaluation_config()
        self.train_config = self.config.get_training_config()
        self.temp_config = self.config.get_temperature_tuning_config()
        self.data_ingestion_config = self.config.get_data_ingestion_config()

    def main(self):
        # Load split data
        df_path = os.path.join(self.data_ingestion_config.unzip_dir, "train_split.csv")
        labels_path = os.path.join(self.data_ingestion_config.unzip_dir, "train_labels.csv")

        df = pd.read_csv(df_path, index_col="id")
        labels_df = pd.read_csv(labels_path, index_col="id")
        df["label"] = labels_df.loc[df.index].values.argmax(1)

        train_df = df[df["split"] == "train"]
        val_df = df[df["split"] == "val"]

        _, val_loader = get_dataloaders(
            train_df= train_df,
            val_df=val_df,
            batch_size=self.train_config.batch_size,
            log_dir=self.train_config.log_dir
        )

        # Load model
        if self.train_config.model_to_use.lower() == "customcnn":
            model = CustomCNN(
                dropout=self.train_config.dropout,
                hidden_units=self.train_config.hidden_units,
                num_classes=self.train_config.num_classes
            )
            model.load_state_dict(torch.load(self.train_config.customcnn_trained_model_path, map_location="cpu"))
        else:
            model = ResNet50Model(self.config.get_resnet_model_config()).get_model()
            model.load_state_dict(torch.load(self.train_config.resnet_trained_model_path, map_location="cpu"))

        evaluator = ModelEvaluator(
            model=model,
            dataloader=val_loader,
            config=self.eval_config,
            temp_config=self.temp_config
        )

        report = evaluator.evaluate()
        logger.info(report)


if __name__ == "__main__":
    try:
        logger.info(f"🔁 STARTING {STAGE_NAME}")
        pipeline = ModelEvaluationPipeline()
        pipeline.main()
        logger.info(f"✅ COMPLETED {STAGE_NAME}")
    except Exception as e:
        logger.exception(f"{STAGE_NAME} FAILED due to: {e}")
        raise e
