# ecoclassify/components/fine_tuning.py

import os
import json
import copy
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from ecoclassify import logger


class FineTuner:
    """
    Fine-tune the already trained ResNet50 on a dataset with the *same classes*.
    - Uses the same normalization stats as training if available.
    - Freezes backbone except layer4 + fc.
    - Early stopping + best weights.
    - Saves to config.output_model_path / output_label_mapping_path.
    """

    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Output dirs from config
        Path(self.config.output_model_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.config.output_label_mapping_path).parent.mkdir(parents=True, exist_ok=True)

        # Mean/std paths to check
        self.mean_std_path_candidates = [
            "artifacts/training/logs/mean_std.json",  
            getattr(self.config, "mean_std_path", ""),  
        ]

    # ---------- utils ----------

    def _load_mean_std(self):
        for p in self.mean_std_path_candidates:
            if p and os.path.exists(p):
                try:
                    with open(p, "r") as f:
                        stats = json.load(f)
                    mean, std = stats["mean"], stats["std"]
                    logger.info(f"✅ Using training stats mean={mean}, std={std}")
                    return mean, std
                except Exception as e:
                    logger.warning(f"Couldn't read mean/std from {p}: {e}")
        logger.warning("⚠️ mean_std.json not found. Falling back to ImageNet stats.")
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    def _build_transforms(self):
        mean, std = self._load_mean_std()

        aug_list = [transforms.RandomResizedCrop(self.config.crop_size)]

        if self.config.flip:
            aug_list.append(transforms.RandomHorizontalFlip())

        if any([self.config.brightness, self.config.contrast,
                self.config.saturation, self.config.hue]):
            aug_list.append(
                transforms.ColorJitter(
                    brightness=self.config.brightness,
                    contrast=self.config.contrast,
                    saturation=self.config.saturation,
                    hue=self.config.hue
                )
            )

        aug_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        train_tf = transforms.Compose(aug_list)
        val_tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.config.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        return train_tf, val_tf

    def _check_labels(self, dataset_classes):
        """Ensure fine-tune dataset classes match the original model's label mapping order."""
        label_map_path = str(self.config.label_mapping_path)  # <-- use config path
        if not os.path.exists(label_map_path):
            logger.warning(f"⚠️ Label mapping not found at {label_map_path}. Skipping label check.")
            return True, dataset_classes

        with open(label_map_path, "r") as f:
            original_map = json.load(f)

        # idx->name mapping may be saved with string keys, normalize:
        if all(str(k).isdigit() for k in original_map.keys()):
            # Keys are numeric
            original_idx_to_name = {int(k): v for k, v in original_map.items()}
        else:
            # Keys are class names, values are numeric IDs
            original_idx_to_name = {v: k for k, v in original_map.items()}
        original_classes_ordered = [original_idx_to_name[i] for i in sorted(original_idx_to_name.keys())]

        if sorted(dataset_classes) != sorted(original_classes_ordered):
            logger.error("❌ Label mismatch! Fine-tuning dataset must contain EXACTLY the same classes.")
            logger.error(f"Expected set: {sorted(original_classes_ordered)}")
            logger.error(f"Found set:    {sorted(dataset_classes)}")
            return False, dataset_classes

        logger.info("✅ Label check passed (same class set).")
        return True, original_classes_ordered  # preserve original order

    def _build_model(self, num_classes: int, unfreeze_backbone: bool = False):
        model = models.resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

        if os.path.exists(self.config.model_path):
            try:
                state = torch.load(self.config.model_path, map_location="cpu")
                model.load_state_dict(state, strict=False)
                logger.info(f"📥 Loaded weights from {self.config.model_path} (strict=False).")
            except Exception as e:
                logger.warning(f"⚠️ Failed loading weights from {self.config.model_path}: {e}")
        else:
            logger.warning(f"⚠️ Model file not found at {self.config.model_path}, starting from scratch.")

        # Default freeze
        for p in model.parameters():
            p.requires_grad = False
        for p in model.layer4.parameters():
            p.requires_grad = True
        for p in model.fc.parameters():
            p.requires_grad = True

        # If user requests full unfreeze
        if unfreeze_backbone:
            logger.info("🔓 Unfreezing full backbone for deeper fine-tuning...")
            for p in model.parameters():
                p.requires_grad = True

        return model

    # ---------- main ----------

    def run(self, callback=None):
        logger.info("🔄 Building datasets with augmentations...")
        train_tf, val_tf = self._build_transforms()

        train_ds = datasets.ImageFolder(self.config.train_dir, transform=train_tf)
        val_ds = datasets.ImageFolder(self.config.val_dir, transform=val_tf)

        ok, ordered_classes = self._check_labels(train_ds.classes)
        if not ok:
            return 0.0

        train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=self.config.batch_size, shuffle=False, num_workers=2, pin_memory=True)

        model = self._build_model(len(ordered_classes), getattr(self.config, "unfreeze_backbone", False)).to(self.device)

        params = [p for p in model.parameters() if p.requires_grad]
        lr = min(self.config.learning_rate, 1e-3)
        optimizer = optim.Adam(params, lr=lr)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=self.config.scheduler_factor,
            patience=self.config.scheduler_patience
        )

        criterion = nn.CrossEntropyLoss()

        best_acc = 0.0
        wait = 0
        best_wts = copy.deepcopy(model.state_dict())

        # Track metrics for potential visualizations
        train_losses = []
        val_losses = []
        val_accuracies = []

        logger.info(f"🚀 Starting fine-tuning for {self.config.epochs} epochs (lr={lr})...")
        for epoch in range(self.config.epochs):
            model.train()
            running_loss = 0.0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / max(1, len(train_loader))
            train_losses.append(avg_loss)

            # Validation
            model.eval()
            val_loss = 0.0
            correct, total = 0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, preds = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (preds == labels).sum().item()

            avg_val_loss = val_loss / max(1, len(val_loader))
            val_acc = correct / max(1, total)
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_acc)

            scheduler.step(val_acc)

            logger.info(f"Epoch [{epoch+1}/{self.config.epochs}]  "
                        f"loss: {avg_loss:.4f}  val_loss: {avg_val_loss:.4f}  val_acc: {val_acc*100:.2f}%")

            # Call the callback if provided (for Streamlit progress/logs)
            if callback is not None:
                try:
                    callback(epoch, self.config.epochs, avg_loss, avg_val_loss, val_acc)
                except Exception as e:
                    logger.warning(f"⚠️ Callback error: {e}")

            if val_acc > best_acc:
                best_acc = val_acc
                best_wts = copy.deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= self.config.patience:
                    logger.info("⏹ Early stopping triggered.")
                    break

        model.load_state_dict(best_wts)
        torch.save(model.state_dict(), str(self.config.output_model_path))
        with open(self.config.output_label_mapping_path, "w") as f:
            json.dump({i: cls for i, cls in enumerate(ordered_classes)}, f)

        logger.info(f"✅ Fine-tuned model saved to: {self.config.output_model_path}")
        logger.info(f"✅ Label mapping saved to:   {self.config.output_label_mapping_path}")

        if callback is None:
            return best_acc
        else:
            return best_acc, train_losses, val_losses, val_accuracies
