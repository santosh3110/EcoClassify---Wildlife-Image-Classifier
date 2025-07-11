from ecoclassify import logger
from ecoclassify.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline

# ====================== STAGE 01 ====================== #
STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f"\n\n>>>>> STARTING {STAGE_NAME} <<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>> COMPLETED {STAGE_NAME} <<<<< \n{'x='*30}")
except Exception as e:
    logger.exception(f"{STAGE_NAME} FAILED due to: {e}")
    raise e
