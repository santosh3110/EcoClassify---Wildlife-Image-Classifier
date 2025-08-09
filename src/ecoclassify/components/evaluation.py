import json
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, log_loss, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import mlflow
from ecoclassify.config.configuration import EvaluationConfig, TemperatureTuningConfig
from ecoclassify import logger


class ModelEvaluator:
    def __init__(self, model, dataloader, config: EvaluationConfig, temp_config: TemperatureTuningConfig,device=None):
        self.model = model
        self.dataloader = dataloader
        self.config = config
        self.temp_config = temp_config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = config.class_names
        

    def _get_logits_labels(self):
        self.model.to(self.device)
        self.model.eval()

        logits_list = []
        labels_list = []

        with torch.no_grad():
            for images, labels in tqdm(self.dataloader, desc="🔍 Collecting logits"):
                images = images.to(self.device)
                outputs = self.model(images)
                logits_list.append(outputs.cpu())
                labels_list.append(labels)

        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)

        return logits, labels

    def _evaluate_metrics(self, preds, labels, probs=None):
        metrics = {
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, average="weighted", zero_division=0),
            "recall": recall_score(labels, preds, average="weighted", zero_division=0),
            "f1_score": f1_score(labels, preds, average="weighted", zero_division=0),
        }

        if probs is not None:
            metrics["log_loss"] = log_loss(labels, probs)

        return metrics

    def _find_best_temperature(self, logits, labels):
        best_T = 1.0
        best_loss = float('inf')
        T_range = self.temp_config.search_range
        steps = self.temp_config.search_steps
        Ts = np.linspace(T_range[0], T_range[1], steps)

        labels_np = labels.numpy()

        for T in Ts:
            scaled_probs = torch.softmax(logits / T, dim=1).numpy()
            try:
                loss = log_loss(labels_np, scaled_probs)
                if loss < best_loss:
                    best_loss = loss
                    best_T = T
            except:
                continue

        return best_T, best_loss

    def _plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.class_names)
        disp.plot(xticks_rotation=90, cmap='Blues')
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(self.config.confusion_matrix_path)
        plt.close()
        logger.info(f"📊 Confusion matrix saved to {self.config.confusion_matrix_path}")

    def evaluate(self):
        logits, labels = self._get_logits_labels()
        labels_np = labels.numpy()

        # ======= Uncalibrated Metrics =======
        uncalibrated_probs = torch.softmax(logits, dim=1).numpy()
        uncalibrated_preds = np.argmax(uncalibrated_probs, axis=1)
        uncalibrated_metrics = self._evaluate_metrics(uncalibrated_preds, labels_np, uncalibrated_probs)

        # ======= Temperature Tuning =======
        if self.temp_config.enabled:
            T, calibrated_log_loss = self._find_best_temperature(logits, labels)
        else:
            T = 1.0
            calibrated_log_loss = None

        calibrated_probs = torch.softmax(logits / T, dim=1).numpy()
        calibrated_preds = np.argmax(calibrated_probs, axis=1)
        calibrated_metrics = self._evaluate_metrics(calibrated_preds, labels_np, calibrated_probs)

        # ======= Confusion Matrix =======
        self._plot_confusion_matrix(labels_np, calibrated_preds)

        # ======= Save Report =======
        report = {
            "temperature": T,
            "uncalibrated": uncalibrated_metrics,
            "calibrated": calibrated_metrics,
            "full_report": classification_report(labels_np, calibrated_preds, target_names=self.class_names, output_dict=True)
        }

        with open(self.config.report_path, "w") as f:
            json.dump(report, f, indent=4)

        logger.info(f"✅ Evaluation report (with temperature tuning) saved to: {self.config.report_path}")

        # ======= Log to MLflow/DAGsHub =======
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        mlflow.set_experiment(f"eval_{self.config.model_to_use}")

        with mlflow.start_run(run_name="evaluation"):
            mlflow.log_metrics(calibrated_metrics)
            mlflow.log_artifact(self.config.report_path)
            mlflow.log_artifact(self.config.confusion_matrix_path)
            
        return report
