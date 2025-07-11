import os
from box.exceptions import BoxValueError
import yaml
from ecoclassify import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any, List, Union
import base64



@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads a YAML file and returns content as ConfigBox."""
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file loaded: {path_to_yaml}")
            return ConfigBox(content)
    except BoxValueError:
        logger.error(f"YAML file is empty: {path_to_yaml}")
        raise ValueError("YAML file is empty.")
    except FileNotFoundError:
        logger.error(f"YAML file not found: {path_to_yaml}")
        raise
    except Exception as e:
        logger.exception(f"Error loading YAML: {e}")
        raise


@ensure_annotations
def create_directories(path_to_directories: list, verbose: bool = True):
    """Creates multiple directories if they don't exist."""
    try:
        for path in path_to_directories:
            os.makedirs(path, exist_ok=True)
            if verbose:
                logger.info(f"Created directory at: {path}")
    except Exception as e:
        logger.exception(f"Failed to create directories: {e}")
        raise


@ensure_annotations
def save_json(path: Path, data: dict):
    """Saves a dictionary to a JSON file."""
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        logger.info(f"JSON saved to: {path}")
    except Exception as e:
        logger.exception(f"Failed to save JSON to {path}: {e}")
        raise


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """Loads a JSON file and returns a ConfigBox."""
    try:
        with open(path) as f:
            content = json.load(f)
        logger.info(f"JSON loaded from: {path}")
        return ConfigBox(content)
    except Exception as e:
        logger.exception(f"Failed to load JSON from {path}: {e}")
        raise


@ensure_annotations
def save_bin(data: Any, path: Path):
    """Saves any object as a binary file using joblib."""
    try:
        joblib.dump(value=data, filename=path)
        logger.info(f"Binary file saved at: {path}")
    except Exception as e:
        logger.exception(f"Failed to save binary file: {path}")
        raise


@ensure_annotations
def load_bin(path: Path) -> Any:
    """Loads a joblib binary file."""
    try:
        data = joblib.load(path)
        logger.info(f"Binary file loaded from: {path}")
        return data
    except Exception as e:
        logger.exception(f"Failed to load binary file: {path}")
        raise


@ensure_annotations
def get_size(path: Path) -> str:
    """Returns size of a file in KB."""
    try:
        size_in_kb = round(os.path.getsize(path) / 1024)
        return f"~ {size_in_kb} KB"
    except Exception as e:
        logger.exception(f"Failed to get size of file: {path}")
        raise


def decodeImage(imgstring: str, fileName: str):
    """Decodes base64 image and saves it to file."""
    try:
        imgdata = base64.b64decode(imgstring)
        with open(fileName, 'wb') as f:
            f.write(imgdata)
        logger.info(f"Image decoded and saved to: {fileName}")
    except Exception as e:
        logger.exception(f"Failed to decode and save image: {e}")
        raise


def encodeImageIntoBase64(croppedImagePath: str) -> bytes:
    """Encodes image from path into base64."""
    try:
        with open(croppedImagePath, "rb") as f:
            encoded = base64.b64encode(f.read())
        logger.info(f"Image encoded from: {croppedImagePath}")
        return encoded
    except Exception as e:
        logger.exception(f"Failed to encode image: {e}")
        raise