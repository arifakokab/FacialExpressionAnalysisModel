# Emotion Recognition System – Facial Expression Analysis (AAI-540 Final Project)

This repository implements the complete MLOps pipeline for a facial expression analysis (FEA) system developed as the final project for AAI-540 (University of San Diego, MSc Applied AI). The solution demonstrates the engineering, deployment, and validation of a MobileNetV2-based emotion recognition model for neuromarketing use cases using AWS SageMaker, batch transform and local batch inference (as batch transform was too costly).

---

## Repository Structure

```
FacialExpressionAnalysisModel/
│
├── Model_Artifacts/                # Trained model files (PyTorch .pth, .tar.gz)
│   ├── mobV2_full.pth
│   └── mobV2_model.tar.gz
│
├── Model_Registry/                 # Model registry outputs (group, package, card)
│   ├── Model Group, Package and Card.ipynb
│   └── model_card.json
│
├── Model_Training/                 # Notebooks and code for model training
│   ├── FEA_MobileNetV2_FER+,RAF-DB.ipynb
│   └── FEA MobileNetV2 FER+,RAF-DB Final Code Final Project.pdf
│
├── Notebooks AWS Jupyter/          # Notebooks for data engineering, inference, etc.
│   ├── 01_extract_and_upload_video_frames.ipynb
│   ├── 02_Run Batch Transform Job.ipynb
│   ├── 03_Local Batch Inference.ipynb
│   ├── Model Creation in SageMaker.ipynb
│   └── inference.py
│
├── README.md                       # Project documentation
│
├── requirements.txt                # Python dependencies
│
└── (Sample output files will be generated during the demo)
```

---

## Components

* **Data Sources:** FER+ and RAF-DB datasets (downloaded from Kaggle)
* **Model:** MobileNetV2 (fine-tuned for 7-class facial emotion recognition)
* **Deployment:** Batch inference (local, with AWS SageMaker Model Registry for artifact management)
* **Monitoring:** Basic evaluation metrics and result analysis
* **CI/CD:** Manual workflow (version-controlled in GitHub)

---

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/arifakokab/FacialExpressionAnalysisModel.git
cd FacialExpressionAnalysisModel
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Model Weights

The trained model checkpoint `mobV2_full.pth` is provided in `Model_Artifacts/`.

### 4. Prepare Input Video

* Place your `.mp4` video file in the project directory.
* Use `01_extract_and_upload_video_frames.ipynb` to extract frames (1 frame per second) from the video.

### 5. Local Batch Inference

* Run `03_Local Batch Inference.ipynb` to load the trained model, process frames, and generate per-second emotion predictions (`emotion_predictions_with_timestamps.csv`).

### 6. Analyze Output

* Open the generated CSV file with Excel or pandas for analysis.

---

## Example Output

| frame           | timestamp | emotion  |
| --------------- | --------- | -------- |
| frame\_0000.jpg | 0:00:00   | Neutral  |
| frame\_0001.jpg | 0:00:01   | Happy    |
| frame\_0002.jpg | 0:00:02   | Surprise |

*Note: Sample output will be generated live in the demo video. It is not pre-uploaded to the repository.*

---

## Project Details

* **Model:** MobileNetV2, fine-tuned for 7 basic emotions
* **Training Data:** FERPlus, RAF-DB (public datasets)
* **Features:** Standard image preprocessing, augmentation, and label smoothing
* **Deployment:** Local batch inference (with optional AWS Batch Transform)
* **Monitoring:** Validation accuracy and macro F1-score, with per-batch output analysis for drift
* **Model Registry:** Model group, package, and card (see `Model_Registry/`)
* **CI/CD:** Codebase and artifacts are version-controlled in GitHub for full traceability

---

## Future Improvements

* Enable automated batch transform and real-time endpoint deployment on AWS
* Build a web UI (e.g., Streamlit) for uploading and visualizing results
* Integrate advanced model/data monitoring (e.g., drift detection, fairness metrics)
* Extend for multi-modal emotion recognition (e.g., EEG, GSR)

---

## Author

**Arifa Kokab**
MSc Applied Artificial Intelligence
University of San Diego

> Final project for AAI-540 – Machine Learning Operations

---

## Documentation

* See `Model_Registry/model_card.json` for detailed model specifications and MLOps metadata.
* Notebooks are annotated with stepwise explanations and sample commands for reproducibility.

---

## Notes

* All code, artifacts, and documentation are organized for ease of reproducibility and team audit.
* Sample outputs (such as the emotion predictions CSV) will be shown live during the demo recording and are not uploaded to the repository for privacy and storage efficiency.
* For additional details, refer to notebook documentation.

