from ecoclassify.config.configuration import ConfigurationManager
from ecoclassify.components.explanation_generator import ExplanationGenerator
from ecoclassify.utils.logger import get_logger

STAGE_NAME = "Generate Explanations Stage"
logger = get_logger(STAGE_NAME.replace(" ", "_").lower())

class ExplanationPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_explanation_config()

    def main(self):
        generator = ExplanationGenerator(config=self.config)
        generator.run()

if __name__ == "__main__":
    try:
        logger.info(f"\n\n STARTED: {STAGE_NAME}")
        pipeline = ExplanationPipeline()
        pipeline.main()
        logger.info(f"COMPLETED: {STAGE_NAME} \n{'='*60}")
    except Exception as e:
        logger.exception(f"{STAGE_NAME} FAILED due to: {e}")
        raise
