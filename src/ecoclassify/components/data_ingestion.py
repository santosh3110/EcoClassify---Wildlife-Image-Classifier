import os
import pandas as pd
import numpy as np
import urllib.request
import zipfile
from sklearn.model_selection import train_test_split
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

    def create_train_val_split(self, unzip_dir, seed=42):
        train_df = pd.read_csv(os.path.join(unzip_dir, "train_features.csv"), index_col="id")
        label_df = pd.read_csv(os.path.join(unzip_dir, "train_labels.csv"), index_col="id")

        # Convert one-hot to class label string
        label_df["label"] = label_df.idxmax(axis=1)
        df = train_df.copy()
        df["label_str"] = label_df["label"]

        # 🔁 Encode class strings to integers
        class_names = sorted(df["label_str"].unique())
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        df["label"] = df["label_str"].map(class_to_idx)

        # 💾 Save label mapping for future use
        label_map_path = os.path.join(unzip_dir, "label_mapping.json")
        with open(label_map_path, "w") as f:
            import json
            json.dump(class_to_idx, f, indent=4)

        # Add filepath (absolute or relative to root)
        df["filepath"] = df["filepath"].apply(lambda x: os.path.join(unzip_dir, x))

        # 🔪 Train/Val split
        train_df, val_df = train_test_split(
            df,
            test_size=0.2,
            stratify=df["label"],
            random_state=seed
        )

        train_df["split"] = "train"
        val_df["split"] = "val"

        combined_df = pd.concat([train_df, val_df])
        combined_df.to_csv(os.path.join(unzip_dir, "train_split.csv"))

        print("✅ Split CSV saved.")
        print(f"📦 Label mapping saved to: {label_map_path}")




    def run(self):
        self.download_file()
        self.extract_zip_file()
        self.create_train_val_split(self.config.unzip_dir)
