from ecoclassify.config.configuration import ConfigurationManager
from ecoclassify.components.batch_inference import BatchInference
from ecoclassify.utils.logger import get_logger

STAGE_NAME = "Batch Inference Stage"
logger = get_logger(STAGE_NAME.replace(" ", "_").lower())

class BatchInferencePipeline:
    def __init__(self):
        self.config = ConfigurationManager()
        self.batch_inference_config = self.config.get_batch_inference_config()

    def main(self):
        batch_inf = BatchInference(self.batch_inference_config)
        batch_inf.run()

if __name__ == "__main__":
    try:
        logger.info(f"STARTED: {STAGE_NAME}")
        obj = BatchInferencePipeline()
        obj.main()
        logger.info(f"COMPLETED: {STAGE_NAME}")
    except Exception as e:
        logger.exception(e)
        raise