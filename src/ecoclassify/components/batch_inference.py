import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import models, transforms
from PIL import Image
import json
from ecoclassify import logger
from ecoclassify.components.data_loader import WildlifeDataset

class BatchInference:
    def __init__(self, config):
        self.config = config

        with open(self.config.label_mapping_path, "r") as f:
            label_map = json.load(f)

        # Ensure correct index → name mapping
        self.idx_to_class = {int(k): v for k, v in label_map.items()}
        self.class_names = [self.idx_to_class[i] for i in sorted(self.idx_to_class.keys())]

    def load_model(self):
        logger.info("🔄 Loading trained model...")
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(self.class_names))
        model.load_state_dict(torch.load(self.config.model_path, map_location="cpu"))
        model.eval()
        return model

    def get_transform(self):
        with open(self.config.mean_std_path, "r") as f:
            stats = json.load(f)
        mean, std = stats["mean"], stats["std"]
        logger.info(f"Using mean: {mean}, std: {std} for normalization")

        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def run(self):
        os.makedirs(self.config.root_dir, exist_ok=True)

        df_test = pd.read_csv(self.config.test_csv)
        test_dir = os.path.dirname(self.config.test_csv)
        df_test["filepath"] = df_test["filepath"].apply(
            lambda p: os.path.join(test_dir, p) if not os.path.isabs(p) else p
        )

        transform = self.get_transform()
        test_dataset = WildlifeDataset(df_test, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size,
                                 shuffle=False, num_workers=self.config.num_workers)

        model = self.load_model()
        results = []

        logger.info("🚀 Running batch inference...")
        with torch.no_grad():
            for images, labels, img_ids in test_loader:
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()

                for img_id, prob_vector in zip(img_ids, probs):
                    row = {"id": img_id}
                    for class_name, prob in zip(self.class_names, prob_vector):
                        row[class_name] = prob
                    results.append(row)

        df_out = pd.DataFrame(results)
        output_path = os.path.join(self.config.root_dir, "batch_predictions.csv")
        df_out.to_csv(output_path, index=False)
        logger.info(f"✅ Predictions saved at: {output_path}")