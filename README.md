# Emotion Recognition System (Final Project for AAI-540)

This repository contains the full MLOps pipeline for my facial expression analysis (FEA) project, including data engineering, model training, deployment, and system validation using AWS SageMaker.

## Components
- **Data Source**: FER+ and RAF-DB (downloaded from Kaggle)
- **Model**: MobileNetV2 fine-tuned on facial expression classes
- **Deployment**: Batch inference via SageMaker
- **Monitoring**: Basic logs, metrics, and evaluation reports
- **CI/CD**: Manual workflow for demonstration

## Quick Start
- `notebooks/` contains JupyterLab notebooks used in SageMaker.
- `models/` contains trained model weights saved in S3.
- `demo/final_demo_video.mp4` is a screen recording showing system operation.

## Team Members
- Arifa Kokab

## Made for final project
- University of San Diego, MSc.AAI AAI-540

See `model_registry/model_card_mobFER.md` for detailed model specifications.
