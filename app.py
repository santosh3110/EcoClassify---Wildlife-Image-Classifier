import os 
import json
import sys
import tempfile
import pandas as pd
import torch
import zipfile
import streamlit as st
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from ecoclassify import logger
from ecoclassify.config.configuration import ConfigurationManager
from ecoclassify.components.batch_inference import BatchInference
from ecoclassify.components.explanation_generator import ExplanationGenerator
from ecoclassify.components.fine_tuning import FineTuner
from ecoclassify.utils.common import load_json

# -------------------
# CONFIG
# -------------------
CONFIG = {
    "model_path": "artifacts/training/resnet_model.pth",
    "label_mapping_path": "artifacts/data_ingestion/extracted_data/label_mapping.json",
    "mean_std_path": "artifacts/training/logs/mean_std.json",
    "gradcam_target_layer": "layer4"  # For ResNet50 last conv block
}

# -------------------
# HELPER FUNCTIONS
# -------------------
@st.cache_resource
def load_model_and_transforms():
    # Load label mapping
    with open(CONFIG["label_mapping_path"], "r") as f:
        label_map = json.load(f)
    if all(str(k).isdigit() for k in label_map.keys()):
        # Keys are numeric
        idx_to_class = {int(k): v for k, v in label_map.items()}
    else:
        # Keys are class names, values are numeric IDs
        idx_to_class = {v: k for k, v in label_map.items()}

    # Load mean/std
    stats = load_json(Path(CONFIG["mean_std_path"]))  # FIXED: ensure Path type
    mean, std = stats["mean"], stats["std"]

    # Model
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(idx_to_class))
    model.load_state_dict(torch.load(CONFIG["model_path"], map_location="cpu"))
    model.eval()

    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return model, transform, idx_to_class, mean, std

def predict_image(model, transform, idx_to_class, image):
    img_t = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.softmax(outputs, dim=1).squeeze().tolist()
    pred_idx = torch.argmax(outputs, dim=1).item()
    return idx_to_class[pred_idx], probs

# -------------------
# STREAMLIT UI
# -------------------
st.set_page_config(page_title="EcoClassify", layout="wide")
st.title("🦉 EcoClassify - Wildlife Image Classifier")

tabs = st.tabs(["ℹ️ About","📸 Inference", "📦 Batch Inference", "🔧 Fine-tuning"])

# -------------------
# TAB 1: ABOUT
# -------------------
with tabs[0]:
    st.header("🌍 About EcoClassify")
    
    st.markdown("""
    **Welcome to EcoClassify!** 🦉  
    Where **wildlife meets deep learning**.  

    **EcoClassify - Wildlife Image Classifier** was born out of the need to **help researchers, educators, and nature lovers** 
    quickly identify species captured in camera trap images — without needing to be a machine learning wizard.
    This project was developed as part of my internship at **[Euron](https://euron.one/)**, with heartfelt thanks to **Sudhanshu Kumar**, Director of Euron, for his guidance and support.

    ---
    ### 🚀 What it does
    1. **Classifies wildlife images** into 8 wildlife specieseight categories — Antelope_Duiker, Bird, Civet_Genet, Hog, Leopard, Monkey_Prosimian, Rodent, 
                and yes… it can even recognize when there’s nothing there at all — just Blank 🙈.
    2. Uses **transfer learning with ResNet50** for strong, accurate predictions.
    3. **Explains predictions** with Grad-CAM heatmaps — so you can *see what the model sees*.
    4. Supports **batch processing** for test datasets.
    5. Lets you **fine-tune the model** on your own custom dataset, straight from the UI.
    
    ---
    ### 🛠 Under the Hood
    - **Frontend:** Streamlit — keeping it simple & interactive.
    - **Model:** PyTorch ResNet50, fine-tuned on a Wildlife Dataset collected from **drivendata.org**.
    - **Image Processing:** OpenCV + torchvision for augmentations & preprocessing.
    - **Explainability:** Grad-CAM visualizations via `torchcam`.
    - **Data Handling:** Pandas, NumPy.

    ---
    ### 🦜 Why it matters
    In the field of wildlife conservation, time matters. 
    Camera traps generate *thousands* of images, and manually sorting them 
    is both tedious and error-prone.  
    EcoClassify helps:
    - Researchers: Quickly analyze species distribution.
    - Educators: Teach students how AI models "think".
    - Wildlife enthusiasts: Get AI-powered insights on sightings.
                
    ---
    ### 📚 Dataset Reference
    The Pan African Programme: The Cultured Chimpanzee, Wild Chimpanzee Foundation, DrivenData. (2022).  
    *Conser-vision Practice Area: Image Classification.*  
    Retrieved [July 12, 2025] from  
    [https://www.drivendata.org/competitions/87/competition-image-classification-wildlife-conservation/](https://www.drivendata.org/competitions/87/competition-image-classification-wildlife-conservation/)

    ---

    ---
    **Made with ❤️ for Wildlife & AI by Santosh Kumar Guntupalli.**  
    _Let’s help protect the wild._
    """)

# -------------------
# TAB 2: INFERENCE
# -------------------
with tabs[1]:
    st.header("Single / Multiple Image Prediction with GradCAM")
    uploaded_files = st.file_uploader("Upload image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        model, transform, idx_to_class, mean, std = load_model_and_transforms()
        explainer_config = type("Config", (object,), {
            "model_weights": CONFIG["model_path"],
            "mean_std_path": CONFIG["mean_std_path"],
            "label_mapping_path": CONFIG["label_mapping_path"],
            "gradcam_target_layer": CONFIG["gradcam_target_layer"],
            "root_dir": "artifacts/streamlit_outputs"
        })()
        os.makedirs(explainer_config.root_dir, exist_ok=True)
        explainer = ExplanationGenerator(explainer_config)

        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file).convert("RGB")
            pred_class, probs = predict_image(model, transform, idx_to_class, image)

            st.subheader(f"Prediction: **{pred_class}**")
            st.bar_chart(pd.Series(probs, index=list(idx_to_class.values())))

            # GradCAM overlay using ExplanationGenerator
            img_tensor = transform(image).unsqueeze(0)
            heatmap, _ = explainer.gradcam(model, img_tensor, CONFIG["gradcam_target_layer"])
            plt_obj = explainer.create_side_by_side(image, heatmap, pred_class, "Unknown")

            # Display side-by-side result
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original Image", use_column_width=True)
            with col2:
                import io
                buf = io.BytesIO()
                plt_obj.savefig(buf, format="png", bbox_inches="tight")
                st.image(buf, caption="GradCAM Overlay", use_column_width=True)
                plt_obj.close()

# -------------------
# TAB 3: BATCH INFERENCE
# -------------------
with tabs[2]:
    st.header("Batch Prediction from CSV + Images ZIP")

    uploaded_csv = st.file_uploader("Upload CSV with filepaths", type=["csv"])
    uploaded_zip = st.file_uploader("Upload ZIP containing images", type=["zip"])

    if uploaded_csv and uploaded_zip:

        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.text("📂 Preparing files...")

            # Create a temp dir
            temp_dir = tempfile.mkdtemp()

            # Save and extract ZIP
            zip_path = os.path.join(temp_dir, "images.zip")
            with open(zip_path, "wb") as f:
                f.write(uploaded_zip.read())
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            progress_bar.progress(20)
            status_text.text("✅ Images extracted")

            # Build map of filename -> actual extracted path
            file_map = {}
            for root, dirs, files in os.walk(temp_dir):
                for f in files:
                    file_map[f] = os.path.join(root, f)

            # Read CSV
            df = pd.read_csv(uploaded_csv)
            progress_bar.progress(40)
            status_text.text("📄 CSV loaded")

            # Fix CSV paths to point to extracted folder
            def fix_path(p):
                filename = os.path.basename(p)
                if filename in file_map:
                    return file_map[filename]
                else:
                    raise FileNotFoundError(f"{filename} not found in uploaded ZIP.")

            df["filepath"] = df["filepath"].apply(fix_path)

            # Save updated CSV in temp dir
            temp_csv_path = os.path.join(temp_dir, "input.csv")
            df.to_csv(temp_csv_path, index=False)

            progress_bar.progress(60)
            status_text.text("⚙️ Configuring batch inference...")

            # Config object for BatchInference
            config_obj = type("Config", (object,), {
                "root_dir": temp_dir,
                "model_path": CONFIG["model_path"],
                "label_mapping_path": CONFIG["label_mapping_path"],
                "mean_std_path": CONFIG["mean_std_path"],
                "test_csv": temp_csv_path,
                "batch_size": 16,
                "num_workers": 0
            })()

            # Run inference
            batch_inf = BatchInference(config_obj)
            batch_inf.run()

            progress_bar.progress(90)

            # Show predictions
            pred_path = os.path.join(temp_dir, "batch_predictions.csv")
            if os.path.exists(pred_path):
                progress_bar.progress(100)
                status_text.text("✅ Inference complete!")
                st.success("✅ Predictions complete!")
                st.dataframe(pd.read_csv(pred_path).head())

                with open(pred_path, "rb") as f:
                    st.download_button("Download CSV", f, file_name="batch_predictions.csv")
            
        except Exception as e:
            logger.exception(e)
            st.error(f"Error during batch inference: {str(e)}")
            status_text.text("❌ Error occurred. Please check logs.")
            progress_bar.progress(0)

# -------------------
# TAB 4: FINE-TUNING
# -------------------
with tabs[3]:
    st.header("🛠 Fine-tuning the Trained Model")
    st.markdown(
        "Upload **ImageNet style dataset** (train/ and val/ folders with species subfolders) "
        "to fine-tune the existing ResNet50 model."
    )

    uploaded_zip = st.file_uploader(
        "Upload dataset ZIP file (train/ and val/ inside)",
        type=["zip"]
    )

    # ---- User-selectable hyperparameters ----
    st.subheader("⚙️ Fine-tuning Settings")
    batch_size = st.number_input("Batch size", min_value=1, max_value=128, value=32, step=1)
    epochs = st.number_input("Epochs", min_value=1, max_value=100, value=5, step=1)
    unfreeze_backbone = st.checkbox("Unfreeze Backbone", value=False, help="Unfreeze all layers for deeper fine-tuning.")
    patience = st.number_input("Early Stopping Patience", min_value=1, max_value=20, value=3, step=1)
    learning_rate = st.number_input("Learning rate", min_value=1e-6, max_value=1.0, value=0.001, step=0.0001, format="%.4f")
    scheduler_patience = st.number_input("Scheduler Patience", min_value=1, max_value=10, value=2, step=1)
    scheduler_factor = st.number_input("Scheduler Factor", min_value=0.1, max_value=1.0, value=0.2, step=0.01)
    crop_size = st.number_input("Crop size", min_value=64, max_value=512, value=224, step=1)
    flip = st.checkbox("Random Horizontal Flip", value=True)
    brightness = st.slider("Brightness", 0.0, 1.0, 0.2, 0.05)
    contrast = st.slider("Contrast", 0.0, 1.0, 0.2, 0.05)
    saturation = st.slider("Saturation", 0.0, 1.0, 0.2, 0.05)
    hue = st.slider("Hue", 0.0, 0.5, 0.1, 0.01)

    if uploaded_zip is not None:
        tmp_dir = tempfile.mkdtemp()
        tmp_zip_path = os.path.join(tmp_dir, "dataset.zip")
        with open(tmp_zip_path, "wb") as f:
            f.write(uploaded_zip.getbuffer())

        with zipfile.ZipFile(tmp_zip_path, "r") as zip_ref:
            zip_ref.extractall(tmp_dir)

        # Paths
        train_dir = os.path.join(tmp_dir, "train")
        val_dir = os.path.join(tmp_dir, "val")

        if not (os.path.exists(train_dir) and os.path.exists(val_dir)):
            st.error("❌ Uploaded ZIP must contain 'train/' and 'val/' directories.")
        else:
            st.success("✅ Dataset extracted successfully.")

            # Config loading (modular)
            config_manager = ConfigurationManager()
            finetune_config = config_manager.get_fine_tuning_config(
                train_dir=train_dir,
                val_dir=val_dir,
                batch_size=batch_size,
                unfreeze_backbone=unfreeze_backbone, 
                epochs=epochs,
                patience=patience,
                learning_rate=learning_rate,
                scheduler_patience=scheduler_patience,
                scheduler_factor=scheduler_factor,
                crop_size=crop_size,
                flip=flip,
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue
            )

            if st.button("🚀 Start Fine-tuning"):
                try:
                    # device info
                    device_info = "cuda" if torch.cuda.is_available() else "cpu"
                    st.info(f"💻 Training on: **{device_info.upper()}**")

                    # Progress bar + real-time log placeholders
                    progress_bar = st.progress(0)
                    log_placeholder = st.empty()
                    log_placeholder.text("🔄 Starting fine-tuning...")

                    finetuner = FineTuner(finetune_config)

                    # Hook for live logging
                    def training_callback(epoch, total_epochs, train_loss, val_loss, val_acc):
                        progress = int(((epoch + 1) / total_epochs) * 100)
                        progress_bar.progress(progress)
                        log_placeholder.markdown(
                            f"**Epoch {epoch+1}/{total_epochs}**  "
                            f"Train Loss: `{train_loss:.4f}`  "
                            f"Val Loss: `{val_loss:.4f}`  "
                            f"Val Acc: `{val_acc*100:.2f}%`"
                        )

                    best_acc, train_losses, val_losses, val_accuracies = finetuner.run(callback=training_callback)
                    if best_acc > 0:
                        st.success(f"Fine-tuning complete! Best Validation Accuracy: {best_acc*100:.2f}% ")
                        model_path = finetune_config.output_model_path
                        label_map_path = finetune_config.output_label_mapping_path

                        col1, col2 = st.columns(2)

                        # Loss curves
                        with col1:
                            st.subheader("📉 Loss Curves")
                            loss_df = pd.DataFrame({
                                "Epoch": [str(e) for e in range(1, len(train_losses) + 1)], 
                                "Train Loss": train_losses,
                                "Validation Loss": val_losses
                            })
                            st.line_chart(loss_df, x="Epoch", y=["Train Loss", "Validation Loss"])

                        # Accuracy curves
                        with col2:
                            st.subheader("✅ Validation Accuracy")
                            acc_df = pd.DataFrame({
                                "Epoch": [str(e) for e in range(1, len(val_accuracies) + 1)], 
                                "Validation Accuracy (%)": [v * 100 for v in val_accuracies]
                            })
                            st.line_chart(acc_df, x="Epoch", y="Validation Accuracy (%)")
                        with open(model_path, "rb") as f:
                            st.download_button("⬇️ Download Fine-tuned Model (.pth)", f, file_name="fine_tuned_model.pth")

                        with open(label_map_path, "rb") as f:
                            st.download_button("⬇️ Download Label Mapping (.json)", f, file_name="label_mapping.json")

                    else:
                        st.error("⚠️ Fine-tuning stopped due to label mismatch.")
                except Exception as e:
                    logger.exception(e)
                    st.error(f"Error during fine-tuning: {str(e)}")
