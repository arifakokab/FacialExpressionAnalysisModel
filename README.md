# Facial Expression Analysis Pipeline for Neuromarketing

## Overview

This repository contains the full codebase and workflow for my final ML system project: **Automated Emotion Recognition from Video for Neuromarketing and Consumer Research**.  
The pipeline processes video files, extracts frames, performs batch inference using a MobileNetV2 deep learning model, and outputs per-frame emotion predictions for seven core emotions.  
All components are designed to run in a modular, auditable manner using AWS SageMaker, S3, and Jupyter notebooks.

## Repository Structure

| Notebook | Purpose |
|----------|---------|
| `00_FEA_MobileNetV2_FER+_RAF_DB.ipynb` | Model training and evaluation (in Google Colab, exported for AWS deployment) |
| `01_Model Creation in SageMaker.ipynb` | Model artifact import and registration in SageMaker Model Registry |
| `02_FER_Model_CICD_SageMaker_AAI540.ipynb` | Automated model registration, package and model card creation, and CI/CD automation |
| `03_extract_and_upload_video_frames.ipynb` | Video frame extraction (with OpenCV) and upload to S3 for batch inference |
| `04_Run Batch Transform Job.ipynb` | Running batch inference in SageMaker and viewing batch outputs |
| `05_Model_Monitoring_Performance.ipynb` | Parsing batch outputs, generating performance metrics, and monitoring via CloudWatch |

- All charts, confusion matrices, and visualizations are generated within the notebooks for clarity.
- All code is commented and structured for reproducibility.

## Data Storage

- **Datasets Used:** [FERPlus](https://www.kaggle.com/datasets/subhaditya/fer2013plus), [RAF-DB](https://www.kaggle.com/datasets/shuvoalok/raf-db-dataset)
- **Data Storage:**  
  All raw and processed data (videos, extracted frames, batch outputs) are securely stored in private AWS S3 buckets.  
  - *Input Videos:* `s3://your-bucket/videos/`
  - *Extracted Frames:* `s3://your-bucket/batch_input/`
  - *Inference Outputs:* `s3://your-bucket/batch_output/`

*Note: No raw data is included in this repository. All code assumes access to the relevant S3 buckets as documented above.*

## ML System Design Document

- The full [ML System Design Document](./ML_System_Design_Document.pdf) details the business case, technical design, data sources, security considerations, and future improvements.

## Instructions

1. **Train the model** (if needed) using `00_FEA_MobileNetV2_FER+_RAF_DB.ipynb` in Colab.
2. **Register and deploy the model** in SageMaker using `01_Model Creation in SageMaker.ipynb` and `02_FER_Model_CICD_SageMaker_AAI540.ipynb`.
3. **Extract video frames** with `03_extract_and_upload_video_frames.ipynb`, and upload to S3.
4. **Run batch inference** using `04_Run Batch Transform Job.ipynb`.
5. **Monitor and evaluate** outputs using `05_Model_Monitoring_Performance.ipynb`.

## Requirements

- Python 3.8+
- AWS SageMaker and S3 access
- OpenCV, boto3, torch, timm, matplotlib, pandas, seaborn, scikit-learn

See each notebook’s first cell for the exact `pip install` commands if running locally.

## Teamwork and Collaboration

This project was completed individually by Arifa Kokab.  
All code and design decisions reflect my independent work, as documented in the commit history.

## License

For academic/research demonstration purposes only.  
FERPlus and RAF-DB datasets used under their respective licenses.

---

## Acknowledgments

Special thanks to the University of San Diego AAI-540 Machine Learning Operations course for providing the structure and support for this project.
