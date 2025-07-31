import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from ecoclassify.config.configuration import TrainingConfig, PrepareCallbacksConfig
from ecoclassify import logger
import mlflow
import dagshub


class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, config: TrainingConfig, model_path, experiment_name, callbacks_config: PrepareCallbacksConfig):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.model_path = model_path
        self.experiment_name = experiment_name
        self.callbacks_config = callbacks_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.checkpoint_path = str(callbacks_config.checkpoint_model_filepath)
        self.start_epoch = 0
        self.best_loss = float("inf")

        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)

    def _load_checkpoint_if_available(self):
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
            self.start_epoch = checkpoint["epoch"] + 1
            self.best_loss = checkpoint["best_loss"]
            logger.info(f"🔄 Resuming from checkpoint at epoch {self.start_epoch}")
        else:
            logger.info("🆕 No checkpoint found. Starting from scratch.")

    def train(self):
        self.model.to(self.device)

        self.optimizer = Adam(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.config.scheduler_factor,
            patience=self.config.scheduler_patience
        )

        self._load_checkpoint_if_available()

        # =================== MLflow DAGsHub ===================
        try:
            dagshub.init(self.config.dagshub_repo_owner, self.config.dagshub_repo_name, mlflow=True)
        except Exception as e:
            logger.warning(f"DAGsHub init failed: {e}")
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(run_name=self.config.model_to_use.lower()):
            mlflow.log_params({
                "epochs": self.config.epochs,
                "lr": self.config.lr,
                "batch_size": self.config.batch_size,
                "label_smoothing": self.config.label_smoothing,
                "weight_decay": self.config.weight_decay,
                "grad_clip": self.config.grad_clip
            })

            logger.info("🚀 Starting training...")

            best_model_wts = self.model.state_dict()
            patience_counter = 0

            for epoch in range(self.start_epoch, self.config.epochs):
                self.model.train()
                train_loss, train_correct, total_train = 0.0, 0, 0

                for images, labels in tqdm(self.train_loader, desc=f"Epoch {epoch+1}"):
                    images, labels = images.to(self.device), labels.to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                    self.optimizer.step()

                    train_loss += loss.item()
                    preds = outputs.argmax(1)
                    train_correct += (preds == labels).sum().item()
                    total_train += labels.size(0)

                avg_train_loss = train_loss / len(self.train_loader)
                train_acc = train_correct / total_train

                # =================== Validation ===================
                self.model.eval()
                val_loss, val_correct, total_val = 0.0, 0, 0
                with torch.no_grad():
                    for images, labels in self.val_loader:
                        images, labels = images.to(self.device), labels.to(self.device)
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                        val_loss += loss.item()
                        preds = outputs.argmax(1)
                        val_correct += (preds == labels).sum().item()
                        total_val += labels.size(0)

                avg_val_loss = val_loss / len(self.val_loader)
                val_acc = val_correct / total_val

                self.scheduler.step(avg_val_loss)

                logger.info(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f} | "
                            f"Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.4f}")

                mlflow.log_metrics({
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "lr": self.optimizer.param_groups[0]["lr"]
                }, step=epoch)

                # =================== Save Best ===================
                if avg_val_loss < self.best_loss:
                    self.best_loss = avg_val_loss
                    best_model_wts = self.model.state_dict()
                    torch.save(best_model_wts, self.model_path)
                    logger.info(f"✅ Best model saved to: {self.model_path}")
                    patience_counter = 0
                else:
                    patience_counter += 1
                    logger.info(f"⏳ Patience Counter: {patience_counter}/{self.config.early_stopping_patience}")
                    if patience_counter >= self.config.early_stopping_patience:
                        logger.info("🛑 Early stopping triggered.")
                        break

                # =================== Save Checkpoint ===================
                torch.save({
                    "epoch": epoch,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "scheduler_state": self.scheduler.state_dict(),
                    "best_loss": self.best_loss
                }, self.checkpoint_path)

                logger.info(f"💾 Checkpoint saved at {self.checkpoint_path}")

        self.model.load_state_dict(best_model_wts)
        return self.model
