---
title: EcoClassify - Wildlife Classifier
emoji: 🦁
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: "1.36.0"
app_file: app.py
pinned: false
---

# 🦉 EcoClassify – Wildlife Image Classifier  

![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.36.0-ff69b4.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-red.svg)
![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)

> **AI-powered wildlife conservation.**  
EcoClassify is an **end-to-end computer vision project** that classifies camera trap images into wildlife species using **transfer learning (ResNet50).**  
It includes **model explainability (Grad-CAM)**, **batch inference**, **fine-tuning via Streamlit UI**, and **MLflow/DagsHub integration** for experiment tracking. It was born out of the need to help researchers, educators, and nature lovers quickly identify species without needing to be a machine learning wizard. This project was developed as part of my internship at **Euron**, with heartfelt thanks to **Sudhanshu Kumar, CEO of Euron,** for his guidance and mentorship.

🔗 **Live Demo on Hugging Face**: [![Hugging Face Spaces](https://img.shields.io/badge/Launch%20App-HuggingFace-orange?logo=huggingface)](https://huggingface.co/spaces/santosh3110/Ecoclassify-Wildlife_Classifier)

---

## 📸 App Screenshots  

| Inference (Single Image) | Grad-CAM Explainability |
|---------------------------|--------------------------|
| ![inference](app_inference_tab.png) | ![gradcam](app_gradcam.png) |  

| Batch Inference | Fine-Tuning |
|-----------------|-------------|
| ![batch](app_batch_inf.png) | ![finetune](app_finetune.png) |  

---

## 📖 Table of Contents  

1. [About](#-about)  
2. [Features](#-features)  
3. [Architecture](#-architecture)  
4. [Dataset](#-dataset)  
5. [Installation](#-installation)  
6. [Usage](#-usage)  
7. [Streamlit App](#-streamlit-app)  
8. [Training & Evaluation](#-training--evaluation)  
9. [Explainability](#-explainability)  
10. [Batch Inference](#-batch-inference)  
11. [Fine-Tuning](#-fine-tuning)  
12. [Design Docs](#-design-docs)  
13. [Results](#-results)  
14. [Future Work](#-future-work)  
15. [Acknowledgements](#-acknowledgements)  

---

## 🌍 About  

Camera traps capture **millions of images** in wildlife conservation projects. Manual classification is slow, error-prone, and not scalable.  

**EcoClassify** provides:  
- 🔬 Automated **species classification** (7 classes + Blank).  
- 🖼️ **Explainability dashboard** (Grad-CAM heatmaps).  
- ⚡ **Batch inference** for CSV/ZIP datasets.  
- 🎛️ **Fine-tuning** interface for custom datasets.  
- 📊 **MLflow/DagsHub** experiment logging.  

---

## 🚀 Features  

- ✅ Species classification: *Antelope_Duiker, Bird, Civet_Genet, Hog, Leopard, Monkey_Prosimian, Rodent, Blank*.  
- ✅ **Transfer learning** with ResNet50 backbone.   
- ✅ **Grad-CAM** explainability for predictions.  
- ✅ **Streamlit app** with multiple tabs: Inference, Batch, Fine-tuning.  
- ✅ **Config-driven training** (YAML params & config).  
- ✅ **Experiment tracking** with MLflow + DagsHub.  

---

## 🏗️ Architecture  

### System Architecture  

```mermaid
flowchart TD
    A[User Uploads Image] --> B[Preprocessing & Augmentation]
    B --> C[Model Inference: ResNet50]
    C --> D[Predictions: Species + Confidence ]
    C --> E[Explainability Engine: Grad-CAM]
    D & E --> F[Streamlit Dashboard: Results]
    F --> G[Download CSV / Fine-tune Model]
```

### End-to-End Pipeline  

```mermaid
graph LR
    A[Data Ingestion] --> B[Data Loader]
    B --> C[Training Pipeline with MLflow Logging]
    C --> D[Evaluation of Models with MLflow Logging]
    D --> E[Batch Inference]
    E --> F[Streamlit App]
    F --> G[Model Fine Tuning]
```

### Project Structure  

```
.
EcoClassify---Wildlife-Image-Classifier/
│
├── app.py                     # Streamlit app entry point
├── main.py                    # Orchestrates full training → eval → inference pipeline
├── setup.py                   # Package setup
├── requirements.txt           # Dependencies
├── README.md                  # Documentation
├── LICENSE
├── colab_code.ipynb           # Code for running the repo on Google Colab
├── params.yaml                # Hyperparameters
├── init_project_structure.py  # Script to bootstrap project tree
│
├── artifacts/                 # All experiment outputs
│   ├── base_model/            # Initial CNN model
│   ├── resnet50_model/        # ResNet50 base model
│   ├── training/              # Trained model checkpoints
│   ├── prepare_callbacks/     # Callback checkpoints
│   ├── evaluation/            # Confusion matrices & reports
│   ├── explanations/          # Grad-CAM heatmaps
│   ├── batch_inference/       # Batch predictions
│   ├── data_ingestion/        # Raw & processed datasets
│   └── streamlit_outputs/     # Models & mappings saved from app
│
├── config/
│   └── config.yaml            # Centralized config file
│ 
├── docs/                      # Project Documents
│   ├── PRD.pdf                # Product Requirements & Specification Document
│   ├── HLD.pdf                # High Level Design Document
│   └── LLD.pdf                # Low Level Design Document
│
├── logs/
│   └── running_logs.log       # Pipeline logs
│
├── research/                  # Notebooks for experiments
│   └── experiment.ipynb
│   
│
└── src/ecoclassify/           # Source code (modular package)
    ├── components/            # Core ML components
    │   ├── customcnn_base_model.py
    │   ├── resnet50_model.py
    │   ├── training.py
    │   ├── evaluation.py
    │   ├── explanation_generator.py
    │   ├── fine_tuning.py
    │   ├── batch_inference.py
    │   ├── data_ingestion.py
    │   └── data_loader.py
    │
    ├── pipeline/              # Orchestrated stages
    │   ├── stage_01_data_ingestion.py
    │   ├── stage_02_customcnn_base_model.py
    │   ├── stage_03_resnet_50_model.py
    │   ├── stage_04_model_training.py
    │   ├── stage_05_model_evaluation.py
    │   ├── stage_06_generate_explanations.py
    │   └── stage_07_batch_inference.py
    │
    ├── config/                # Config manager
    │   └── configuration.py
    │
    ├── constants/             # File paths & constants
    │   └── paths.py
    │
    ├── entity/                # Config/data entities
    │   └── config_entity.py
    │
    ├── utils/                 # Utility functions
    │   ├── common.py
    │   └── logger.py
    │
    └── __init__.py
```

---

## 📚 Dataset  

- **Source**: Conser-vision Practice Area: Image Classification by drivendata.org
- **Provided by**:  
  *The Pan African Programme: The Cultured Chimpanzee, Wild Chimpanzee Foundation, DrivenData. (2022). Conser-vision Practice Area: Image Classification. Retrieved [July 12 2025] from https://www.drivendata.org/competitions/87/competition-image-classification-wildlife-conservation/.*  

---

## ⚙️ Installation  

```bash
git clone https://github.com/santosh3110/EcoClassify---Wildlife-Image-Classifier.git
cd EcoClassify---Wildlife-Image-Classifier
conda create -n ecoclassify python=3.10 -y
conda activate ecoclassify
pip install -r requirements.txt
```

(Optional: install PyTorch with CUDA if using GPU).  

---

## ▶️ Usage  

### Run Streamlit App  

```bash
streamlit run app.py
```

App opens at **http://localhost:8501**.  

### CLI Training  

```bash
python ecoclassify/pipelines/main.py
```

---

## 🖥️ Streamlit App  

👉 Try EcoClassify directly without setup: [Live Demo on Hugging Face 🚀](https://huggingface.co/spaces/santosh3110/Ecoclassify-Wildlife_Classifier)

Tabs available:  

1. **About** – Project info, dataset, motivation.  
2. **Inference** – Upload images → classification + Grad-CAM heatmaps.  
3. **Batch Inference** – Upload CSV + ZIP → get predictions CSV.  
4. **Fine-Tuning** – Upload dataset (train/val) → retrain ResNet50 with custom hyperparameters.  

---

## 📊 Model Training & Evaluation  

- Models trained:  
  - **CustomCNN** (100 epochs)  
  - **ResNet50 (transfer learning)** (50 epochs)  

- Evaluation scope:
  - Confusion matrix
  - Classification report
  - Calibration metrics (temperature scaling)
  - **Artifacts** stored under artifacts/

### Results Summary

| Model     | Temperature | Uncalibrated Accuracy | Calibrated Accuracy | Uncalibrated Precision | Calibrated Precision | Uncalibrated Recall | Calibrated Recall | Uncalibrated F1 | Calibrated F1 | Uncalibrated Log-Loss | Calibrated Log-Loss |
|:----------|:-----------:|:---------------------:|:-------------------:|:----------------------:|:--------------------:|:--------------------:|:-------------------:|:----------------:|:----------------:|:----------------------:|:----------------------:|
| CustomCNN | 0.66        | 0.70                  | 0.70                | 0.70                   | 0.70                 | 0.70                 | 0.70                | 0.69             | 0.69          | 0.89                   | 0.83                   |
| ResNet50  | 0.82        | 0.89                  | 0.89                | 0.89                   | 0.89                 | 0.89                 | 0.89                | 0.89             | 0.89          | 0.41                   | 0.39                   |

### Macro & Weighted Averages

#### Macro Averages
| Model     | Macro Precision | Macro Recall | Macro F1 |
|:----------|:----------------:|:------------:|:--------:|
| CustomCNN | 0.7223936480863933 | 0.7029769514860553 | 0.7050137565770959 |
| ResNet50  | 0.8953738491042441 | 0.9009659158187813 | 0.8968504618230253 |

#### Weighted Averages
| Model     | Weighted Precision | Weighted Recall | Weighted F1 |
|:----------|:------------------:|:---------------:|:-----------:|
| CustomCNN | 0.7021639180835437 | 0.6979987871437234 | 0.6919238501157194 |
| ResNet50  | 0.8880514935264254 | 0.8920557913887205 | 0.8886539869936168 |

### Per-class Metrics

#### CustomCNN
| Class            | Precision | Recall | F1-score | Support |
|:----------------:|:---------:|:------:|:--------:|:-------:|
| antelope_duiker  | 0.5020161290322581 | 0.503030303030303 | 0.5025227043390514 | 495.0 |
| bird             | 0.7525083612040134 | 0.6859756097560976 | 0.7177033492822966 | 328.0 |
| blank            | 0.624031007751938 | 0.36343115124153497 | 0.4593437945791726 | 443.0 |
| civet_genet      | 0.8043478260869565 | 0.9154639175257732 | 0.8563162970106075 | 485.0 |
| hog              | 0.9226190476190477 | 0.7948717948717948 | 0.8539944903581267 | 195.0 |
| leopard          | 0.9053117782909931 | 0.8691796008869179 | 0.8868778280542986 | 451.0 |
| monkey_prosimian | 0.5739385065885798 | 0.7871485943775101 | 0.663844199830652  | 498.0 |
| rodent           | 0.6943765281173594 | 0.7047146401985112 | 0.6995073891625616 | 403.0 |

#### ResNet50
| Class            | Precision | Recall | F1-score | Support |
|:----------------:|:---------:|:------:|:--------:|:-------:|
| antelope_duiker  | 0.8137651821862348 | 0.8121212121212121 | 0.8129423660262892 | 495.0 |
| bird             | 0.9349112426035503 | 0.9634146341463414 | 0.948948948948949 | 328.0 |
| blank            | 0.7690140845070422 | 0.6162528216704289 | 0.6842105263157895 | 443.0 |
| civet_genet      | 0.9440000000000000 | 0.9731958762886598 | 0.9583756345177665 | 485.0 |
| hog              | 0.9696969696969697 | 0.9846153846153847 | 0.9770992366412213 | 195.0 |
| leopard          | 0.9444444444444444 | 0.9800443458980045 | 0.9619151251360174 | 451.0 |
| monkey_prosimian | 0.9005847953216374 | 0.927710843373494 | 0.913946587537092  | 498.0 |
| rodent           | 0.8865740740740741 | 0.9503722084367245 | 0.9173652694610779 | 403.0 |

  - Training visuals:
    - CustomCNN Training Chart: ![customcnn_training.png](customcnn_training.png)
    - ResNet50 Training Chart: ![resnet50_training.png](resnet50_training.png)
  - **Hyperparameters** are controlled via the params.yaml file (no hard-coded values)

- Results tracked via ![**MLflow & DagsHub**](https://dagshub.com/santoshkumarguntupalli/EcoClassify---Wildlife-Image-Classifier/experiments)

- Confusion matrices:
 
| CustomCNN confusion matrix| ResNet50 confusion matrix |
|---------------------------|---------------------------|
| ![customcnn_confusion](artifacts/evaluation/customcnn/confusion_matrix.png) | ![resnet50_confusion](artifacts/evaluation/resnet50/confusion_matrix.png)|  

---

## 🔎 Explainability  

- **Grad-CAM** highlights model focus regions.  
- Outputs side-by-side comparison:  
  - Original Image  
  - Heatmap Overlay  
- Sample Grad-CAM heatmaps generated on Val dataset:
    ![alt text](artifacts/explanations/ZJ000039_gradcam.png)
    ![alt text](artifacts/explanations/ZJ003443_gradcam.png)
    ![alt text](artifacts/explanations/ZJ013423_gradcam.png)
    ![alt text](artifacts/explanations/ZJ012512_gradcam.png)
---

## 📦 Batch Inference  

- Upload **CSV** (image paths) + **ZIP** (images).  
- Pipeline produces **predictions.csv** with class & confidence.  

---

## 🛠️ Fine-Tuning  

- Upload dataset in structure:  

```
dataset.zip
 ├── train/
 │   ├── class1/
 │   ├── class2/
 └── val/
     ├── class1/
     ├── class2/
```

- Configure hyperparams (epochs, batch size, LR, early stopping).  
- Retrains ResNet50 on uploaded data.  
- Outputs: new model weights + mapping.  

---

## 📄 Design Docs  

📌 Included in `/docs`:  

- **PRD** – Product Requirements & Specs  
- **HLD** – High-Level Architecture Design  
- **LLD** – Low-Level Implementation Design  

---

## 🧩 Future Work  

- 🚀 Deploy as **FastAPI + Docker** microservice.  
- 📱 Extend to **mobile app** for field researchers.  
- 🧪 Add **ensemble models** (ResNet + ViT).  
- 🐾 Multi-label support (detect multiple species in one frame).  

---

## ❤️ Acknowledgements  

- The Pan African Programme: The Cultured Chimpanzee, Wild Chimpanzee Foundation, DrivenData. (2022). Conser-vision Practice Area: Image Classification. Retrieved [July 12 2025] from https://www.drivendata.org/competitions/87/competition-image-classification-wildlife-conservation/. 
- Mentorship: **Sudhanshu Kumar (Euron)**  
- Frameworks: PyTorch, Streamlit, MLflow, TorchCAM  

---

## 📜 License  

Apache 2.0 License © 2025 Santosh Kumar Guntupalli  

---

✨ *Made with love for Wildlife & AI* 🐆🌱  