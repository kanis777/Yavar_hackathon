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

## Person detection with YOLO 
 <img src="https://github.com/kanis777/Yavar_hackathon/blob/main/output/person.png" alt="Person Predicted" width="400">

## Pose detection with mediapipe
 <img src="https://github.com/kanis777/Yavar_hackathon/blob/main/output/pose.png" alt="Pose detected" width="400">

## fall detection with OWL-ViT
 <img src="https://github.com/kanis777/Yavar_hackathon/blob/main/output/fall.png" alt="Fall detected" width="400">

## Evaluation Criteria

- **High Accuracy:** The model should accurately detect falls in the video sequences.
- **Minimal False Positives:** The solution should minimize false positive detections to avoid unnecessary alarms.
<img src="https://github.com/kanis777/Yavar_hackathon/blob/main/output/metrics.png" alt="Metrics" width="400">
## Usage
  
1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/your-username/yavar-internship-assignment.git```
