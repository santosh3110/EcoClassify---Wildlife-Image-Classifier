from ecoclassify.config.configuration import ConfigurationManager
from ecoclassify.components.data_ingestion import DataIngestion
from ecoclassify.utils.logger import get_logger

STAGE_NAME = "Data Ingestion Stage"
logger = get_logger(STAGE_NAME.replace(" ", "_").lower())


class DataIngestionTrainingPipeline:
    def __init__(self):
        self.config = ConfigurationManager()
        self.data_ingestion_config = self.config.get_data_ingestion_config()

    def main(self):
        ingestion = DataIngestion(config=self.data_ingestion_config)
        ingestion.run()


if __name__ == "__main__":
    try:
        logger.info(f"\n\n STARTED: {STAGE_NAME}")
        pipeline = DataIngestionTrainingPipeline()
        pipeline.main()
        logger.info(f"COMPLETED: {STAGE_NAME} \n{'='*60}")
    except Exception as e:
        logger.exception(f"{STAGE_NAME} FAILED due to: {e}")
        raise
