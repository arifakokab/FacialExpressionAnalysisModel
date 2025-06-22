# Emotion Recognition System (Final Project for AAI-540)

This repository contains the full MLOps pipeline for my facial expression analysis (FEA) project, including data engineering, model training, deployment, and system validation using AWS SageMaker. This project develops and demonstrates a facial expression analysis system using a deep learning model (MobileNetV2) trained on FERPlus and RAF-DB datasets. The system is designed for neuromarketing use cases, enabling marketers and UX researchers to analyze the emotions expressed by participants as they view advertisements or digital content.

---

## Components
- **Data Source**: FER+ and RAF-DB (downloaded from Kaggle)
- **Model**: MobileNetV2 fine-tuned on facial expression classes
- **Deployment**: Batch inference via SageMaker
- **Monitoring**: Basic logs, metrics, and evaluation reports
- **CI/CD**: Manual workflow for demonstration

---

## Quick Start
- `notebooks/` contains JupyterLab notebooks used in SageMaker.
- `models/` contains trained model weights saved in S3.
- `demo/final_demo_video.mp4` is a screen recording showing system operation.

---

- **Input:** Pre-recorded video files
- **Process:** Extract frames, predict per-frame emotion, timestamp results
- **Output:** CSV file with per-second emotion predictions

---

## Repository Structure

├── 01_extract_and_upload_video_frames.ipynb # Extract video frames for inference

├── 02_Run_Batch_Transform_Job.ipynb # Attempted AWS batch transform pipeline

├── 03_Local_Batch_Inference.ipynb # Local batch inference on video frames

├── emotion_predictions_with_timestamps.csv # Sample output: timestamped emotion predictions

├── requirements.txt # Python dependencies

├── README.md # Project documentation (this file)

---

## Environment Setup

1. **Clone this repo:**
    ```bash
    git clone https://github.com/arifakokab/FacialExpressionAnalysisModel.git
    cd FacialExpressionAnalysisModel
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Download Trained Model:**
    - The trained model checkpoint is `mobV2_full.pth` 
  

4. **Prepare Input Video:**
    - Place your MP4 video in the project directory.
    - Use `01_extract_and_upload_video_frames.ipynb` to extract frames at 1 frame/sec.

---

## How to Run

**Step 1: Extract Frames from Video**
- Open and run `01_extract_and_upload_video_frames.ipynb`
- Output will be a folder of frames (e.g., `video_frames/`)

**Step 2: Local Batch Inference**
- Open and run `03_Local_Batch_Inference.ipynb`
- This notebook loads the trained model, processes each frame, and generates `emotion_predictions_with_timestamps.csv`

**Step 3: View Results**
- Open the CSV in your notebook or with Excel/pandas to analyze per-second emotions.

---

## Example Output

| frame         | timestamp | emotion   |
|---------------|-----------|-----------|
| frame_0000.jpg| 0:00:00   | Neutral   |
| frame_0001.jpg| 0:00:01   | Happy     |
| frame_0002.jpg| 0:00:02   | Surprise  |

---

## Project Details Summary

- **Model:** MobileNetV2, fine-tuned for 7 emotions
- **Training Data:** FERPlus, RAF-DB
- **Features:** Standard image preprocessing and augmentation
- **Deployment:** Local inference (AWS SageMaker Model Registry for artifact management)
- **Monitoring:** Distribution of emotion predictions per batch is logged for drift monitoring.  
- **CI/CD:** All code is version-controlled in GitHub. The repo structure supports easy updates and reproducibility.  

---

## Future Work

- Integrate with real-time endpoints or batch transform on AWS
- Build a web UI (e.g., Streamlit app) for user uploads and visualization
- Add more advanced model/data monitoring (e.g., drift detection)
- Extend for multi-modal (EEG, GSR) inputs

---

## Author

**Arifa Kokab**  
MSc Applied Artificial Intelligence

## Made for final project
- University of San Diego, MSc.AAI AAI-540

See `model_registry/model_card_mobFER.md` for detailed model specifications.
