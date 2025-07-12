from ecoclassify import logger
from ecoclassify.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from ecoclassify.pipeline.stage_02_customcnn_base_model import CustomCNNTrainingPipeline

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

# ====================== STAGE 02 ====================== #
STAGE_NAME = "CustomCNN Base Model"

try:
    logger.info(f"\n\n>>>>> STARTING {STAGE_NAME} <<<<<")
    customcnn_base_model = CustomCNNTrainingPipeline()
    customcnn_base_model.main()
    logger.info(f">>>>> COMPLETED {STAGE_NAME} <<<<< \n{'x='*30}")
except Exception as e:
    logger.exception(f"{STAGE_NAME} FAILED due to: {e}")
    raise e