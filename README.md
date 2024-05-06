# Yavar_hackathon

# Yavar Internship Selection Assignment

## Problem Statement

People's fall detection is a critical concern due to its potentially life-threatening consequences, especially in environments such as staircases, escalators, and steps. The assignment requires implementing a solution for fall detection in videos, with specific emphasis on detecting falls accurately and minimizing false positives.

### Libraries/Algorithms Used

- YOLO (You Only Look Once) Computer Vision Library for object detection
- Mediapipe Pose Estimation algorithm for detecting human poses
- OWL-ViT (Open-Vocabulary Vision Transformer) for object detection and recognition
- Scikit-learn for evaluating the model's performance metrics

## Input

Offline video files in mp4 format containing scenes where fall detection needs to be performed.

## Evaluation Criteria

- **High Accuracy:** The model should accurately detect falls in the video sequences.
- **Minimal False Positives:** The solution should minimize false positive detections to avoid unnecessary alarms.

## Output & Timeline

The assignment consists of two deliverables:

1. **Fall Detection Approach Document:** A detailed write-up explaining various possible approaches for fall detection, including recommendations for the selected approach. This document also provides insights into the training phase, suggesting the use of Zero-shot object detection models.
   
2. **Skeletal Solution of Training and Prediction Phases:** A code skeleton for training and prediction phases, showcasing the integration of YOLO, Mediapipe Pose Estimation, and OWL-ViT models for fall detection.

Both deliverables should be uploaded to GitHub, and the link should be provided to the Yavar team by [End of Day (EoD) 6th May 2024].

## Contents

The repository contains the following files:

- `fall_detection_approach_document.pdf`: Detailed documentation explaining the fall detection approach and recommendations.
- `training_and_prediction_skeleton.py`: Python code skeleton for training and prediction phases.
- `requirements.txt`: List of required Python packages and their versions.
- `README.md`: This file, providing an overview of the assignment and its components.

## Usage

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/your-username/yavar-internship-assignment.git
