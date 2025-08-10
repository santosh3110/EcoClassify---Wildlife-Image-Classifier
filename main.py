from ecoclassify import logger
from ecoclassify.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from ecoclassify.pipeline.stage_02_customcnn_base_model import CustomCNNTrainingPipeline
from ecoclassify.pipeline.stage_03_resnet_50_model import Resnet50TrainingPipeline
from ecoclassify.pipeline.stage_04_model_training import ModelTrainingPipeline
from ecoclassify.pipeline.stage_05_model_evaluation import ModelEvaluationPipeline
from ecoclassify.pipeline.stage_06_generate_explanations import ExplanationPipeline
from pathlib import Path

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

# ====================== STAGE 03 ====================== #
STAGE_NAME = "Resnet50 Model"

try:
    logger.info(f"\n\n>>>>> STARTING {STAGE_NAME} <<<<<")
    resnet_50_model = Resnet50TrainingPipeline()
    resnet_50_model.main()
    logger.info(f">>>>> COMPLETED {STAGE_NAME} <<<<< \n{'x='*30}")
except Exception as e:
    logger.exception(f"{STAGE_NAME} FAILED due to: {e}")
    raise e

# ====================== STAGE 04 ====================== #
STAGE_NAME = "Model Training"

try:
    logger.info(f"\n\n>>>>> STARTING {STAGE_NAME} <<<<<")
    model_train_pipeline = ModelTrainingPipeline()
    model_train_pipeline.main()
    logger.info(f">>>>> COMPLETED {STAGE_NAME} <<<<< \n{'x='*30}")
except Exception as e:
    logger.exception(f"{STAGE_NAME} FAILED due to: {e}")
    raise e

# ====================== STAGE 05 ====================== #
STAGE_NAME = "Model Evaluation"

try:
    logger.info(f"\n\n>>>>> STARTING {STAGE_NAME} <<<<<")
    model_evaluation_pipeline = ModelEvaluationPipeline()
    model_evaluation_pipeline.main()
    logger.info(f">>>>> COMPLETED {STAGE_NAME} <<<<< \n{'x='*30}")
except Exception as e:
    logger.exception(f"{STAGE_NAME} FAILED due to: {e}")
    raise e

# ====================== STAGE 06 ====================== #
STAGE_NAME = "Generate Explanations (Grad-CAM)"

try:
    logger.info(f"\n\n>>>>> STARTING {STAGE_NAME} <<<<<")
    explanation_pipeline = ExplanationPipeline()
    explanation_pipeline.main()
    logger.info(f">>>>> COMPLETED {STAGE_NAME} <<<<< \n{'x='*30}")
except Exception as e:
    logger.exception(f"{STAGE_NAME} FAILED due to: {e}")
    raise e

# ====================== STAGE 07 ====================== #
STAGE_NAME = "Batch Inference"  
try:
    logger.info(f"\n\n>>>>> STARTING {STAGE_NAME} <<<<<")
    from ecoclassify.pipeline.stage_07_batch_inference import BatchInferencePipeline
    batch_inference_pipeline = BatchInferencePipeline()
    batch_inference_pipeline.main()
    logger.info(f">>>>> COMPLETED {STAGE_NAME} <<<<< \n{'x='*30}")
except Exception as e:
    logger.exception(f"{STAGE_NAME} FAILED due to: {e}")
    raise e