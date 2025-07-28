from ecoclassify.config.configuration import ConfigurationManager
from ecoclassify.components.resnet50_model import ResNet50Model
from ecoclassify.utils.logger import get_logger

STAGE_NAME = "Resnet50 Model"
logger = get_logger(STAGE_NAME.replace(" ", "_").lower())


class Resnet50TrainingPipeline:
    def __init__(self):
        self.config = ConfigurationManager()
        self.resnet50_model_config = self.config.get_resnet_model_config()
        self.train_config = self.config.get_training_config()

    def main(self):
        resnet50_model = ResNet50Model(config=self.resnet50_model_config, train_config=self.train_config)
        resnet50_model.run()


if __name__ == "__main__":
    try:
        logger.info(f"\n\n STARTED: {STAGE_NAME}")
        pipeline = Resnet50TrainingPipeline()
        pipeline.main()
        logger.info(f"COMPLETED: {STAGE_NAME} \n{'='*60}")
    except Exception as e:
        logger.exception(f"{STAGE_NAME} FAILED due to: {e}")
        raise
