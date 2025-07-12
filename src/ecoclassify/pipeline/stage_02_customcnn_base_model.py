from ecoclassify.config.configuration import ConfigurationManager
from ecoclassify.components.customcnn_base_model import PrepareCustomCNN
from ecoclassify.utils.logger import get_logger

STAGE_NAME = "CustomCNN Base Model"
logger = get_logger(STAGE_NAME.replace(" ", "_").lower())


class CustomCNNTrainingPipeline:
    def __init__(self):
        self.config = ConfigurationManager()
        self.base_model_config = self.config.get_base_model_config()

    def main(self):
        customcnn_base_model = PrepareCustomCNN(config=self.base_model_config)
        customcnn_base_model.run()


if __name__ == "__main__":
    try:
        logger.info(f"\n\n STARTED: {STAGE_NAME}")
        pipeline = CustomCNNTrainingPipeline()
        pipeline.main()
        logger.info(f"COMPLETED: {STAGE_NAME} \n{'='*60}")
    except Exception as e:
        logger.exception(f"{STAGE_NAME} FAILED due to: {e}")
        raise
