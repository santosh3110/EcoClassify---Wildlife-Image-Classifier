import json
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, log_loss, confusion_matrix
from ecoclassify.config.configuration import EvaluationConfig
from ecoclassify import logger
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


class ModelEvaluator:
    def __init__(self, model, dataloader, config: EvaluationConfig, device=None):
        self.model = model
        self.dataloader = dataloader
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _plot_confusion_matrix(self, conf_matrix):
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt=".2f")
        plt.title("Normalized Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        cm_path = os.path.join(os.path.dirname(self.config.root_dir), "confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()
        logger.info(f"🖼️ Confusion matrix saved to: {cm_path}")


    def evaluate(self):
        self.model.to(self.device)
        self.model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(self.dataloader, desc="🔍 Evaluating"):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                preds = outputs.argmax(1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Compute metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
        recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
        f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        logloss = log_loss(all_labels,all_preds)
        class_report = classification_report(all_labels, all_preds, output_dict=True)
        conf_matrix = confusion_matrix(all_labels, all_preds, normalize='true')

        report = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "log loss": logloss,
            "full_report": class_report
        }

        self._plot_confusion_matrix(conf_matrix)

        # Save report
        with open(self.config.report_path, "w") as f:
            json.dump(report, f, indent=4)

        logger.info(f"✅ Evaluation report saved to: {self.config.report_path}")
        return report
