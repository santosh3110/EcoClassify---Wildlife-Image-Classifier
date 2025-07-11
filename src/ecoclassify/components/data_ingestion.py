import os
import urllib.request
import zipfile
from ecoclassify.config.configuration import DataIngestionConfig
from ecoclassify.utils.common import get_size
from ecoclassify import logger
from pathlib import Path

logger.info("data_ingestion")


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            try:
                logger.info("Downloading dataset...")
                _ , headers = urllib.request.urlretrieve(
                    self.config.source_URL,
                    self.config.local_data_file
                )
                logger.info(f"Dataset downloaded with following info: \n{headers}")
            except Exception as e:
                logger.error(f"Failed to download the Dataset: {e}")
                raise
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")

    def extract_zip_file(self):
        unzip_dir = self.config.unzip_dir
        zip_path = self.config.local_data_file

        if not os.path.exists(unzip_dir) or not os.listdir(unzip_dir):
            try:
                logger.info("Extracting zip file...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(unzip_dir)
                logger.info(f"Extracted to: {unzip_dir}")
            except Exception as e:
                logger.error(f"Failed to extract zip: {e}")
                raise
        else:
            logger.info(f"Extraction directory already populated: {unzip_dir}")

    def run(self):
        self.download_file()
        self.extract_zip_file()
