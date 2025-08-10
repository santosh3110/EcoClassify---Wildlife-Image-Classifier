import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms
import pandas as pd
import json
import numpy as np
import os
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from ecoclassify import logger

class ExplanationGenerator:
    def __init__(self, config):
        self.config = config
        os.makedirs(self.config.root_dir, exist_ok=True)

    def load_model(self):
        logger.info("🔄 Loading ResNet50 architecture...")
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 8)  
        model.load_state_dict(torch.load(self.config.model_weights, map_location="cpu"))
        model.eval()
        return model

    def load_mean_std(self):
        with open(self.config.mean_std_path, "r") as f:
            stats = json.load(f)
        return stats["mean"], stats["std"]

    def load_label_mapping(self):
        with open(self.config.label_mapping_path, "r") as f:
            return json.load(f)

    def gradcam(self, model, img_tensor, target_layer):
        activations = {}
        gradients = {}

        def forward_hook(module, inp, out):
            activations["value"] = out.detach()

        def backward_hook(module, grad_in, grad_out):
            gradients["value"] = grad_out[0].detach()

        layer = dict([*model.named_modules()])[target_layer]
        layer.register_forward_hook(forward_hook)
        layer.register_backward_hook(backward_hook)

        pred = model(img_tensor)
        class_idx = pred.argmax(dim=1).item()
        pred[:, class_idx].backward()

        grads = gradients["value"]
        acts = activations["value"]

        pooled_grads = grads.mean(dim=(2, 3), keepdim=True)
        weighted_acts = acts * pooled_grads
        heatmap = weighted_acts.sum(dim=1).squeeze()
        heatmap = np.maximum(heatmap.cpu(), 0)
        heatmap /= torch.max(heatmap)
        return heatmap.numpy(), class_idx

    def overlay_heatmap(self, img_pil, heatmap):
        heatmap = np.uint8(255 * heatmap)
        heatmap = Image.fromarray(heatmap).resize(img_pil.size)
        heatmap = np.array(heatmap)
        plt.imshow(img_pil)
        plt.imshow(heatmap, cmap="jet", alpha=0.4)
        plt.axis("off")

    def create_side_by_side(self, img_pil, heatmap, pred_label, true_label):
        # Resize heatmap to match image
        heatmap_resized = Image.fromarray(np.uint8(255 * heatmap)).resize(img_pil.size)
        heatmap_array = np.array(heatmap_resized)

        # Overlay heatmap on original
        plt.figure(figsize=(8, 4))

        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(img_pil)
        plt.title(f"Original\n(True: {true_label})", fontsize=10)
        plt.axis("off")

        # Grad-CAM overlay
        plt.subplot(1, 2, 2)
        plt.imshow(img_pil)
        plt.imshow(heatmap_array, cmap="jet", alpha=0.4)
        plt.title(f"Grad-CAM (Pred: {pred_label})")
        plt.axis("off")

        return plt


    def run(self):
        logger.info("🚀 Starting explanation generation...")

        model = self.load_model()
        mean, std = self.load_mean_std()
        label_map = self.load_label_mapping()
        idx_to_class = {v: k for k, v in label_map.items()}

        df = pd.read_csv(self.config.train_split_csv)
        val_df = df[df["split"] == "val"].sample(self.config.num_images, random_state=42)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        for _, row in val_df.iterrows():
            img_path = row["filepath"]
            true_label = idx_to_class[row["label"]]
            img_pil = Image.open(img_path).convert("RGB")
            img_tensor = transform(img_pil).unsqueeze(0)

            heatmap, pred_idx = self.gradcam(model, img_tensor, self.config.gradcam_target_layer)
            pred_label = idx_to_class[pred_idx]

            plt_obj = self.create_side_by_side(img_pil, heatmap, pred_label, true_label)
            save_path = Path(self.config.root_dir) / f"{Path(img_path).stem}_gradcam.png"
            plt_obj.savefig(save_path, bbox_inches="tight")
            plt_obj.close()

            logger.info(f"🖼 Saved Grad-CAM: {save_path}")

        logger.info("✅ Explanation generation completed.")