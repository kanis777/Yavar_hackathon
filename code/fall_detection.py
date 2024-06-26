# -*- coding: utf-8 -*-
"""Fall_Detection.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1-g5YLGkP4IeE8GDBZv21_u_00pLlX-Q5
"""

# Step 1: Install required dependencies
!pip install opencv-python-headless
import cv2
import numpy as np
from google.colab.patches import cv2_imshow
import pickle

# Step 2: Download YOLO weights and configuration
!wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov3.weights
!wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
# Download coco.names file
!wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names

# Step 3: Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

from google.colab import drive
drive.mount('/content/drive')

"""YOLO IMPLEMENTED

"""

def resize_frames(frames, target_width=640, target_height=480):
    resized_frames = [cv2.resize(frame, (target_width, target_height)) for frame in frames]
    return resized_frames

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Provide the path to your .mp4 video file
video_path = "/content/drive/MyDrive/Yavar/output_video.mp4"

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video was successfully opened
if not cap.isOpened():
    print("Error: Unable to open the video file. Check the path and permissions.")
else:
    # Get the frames per second (FPS)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Frames per second (FPS): {fps}")

    # Determine the frame interval to achieve a specified time delay (in seconds)
    interval_seconds = 0.5  # Change to display one frame every 0.5 seconds
    frame_interval = int(fps * interval_seconds)  # Frames to skip to achieve the desired interval

    frame_count = 0  # Frame counter

    # Loop to read and display frames at specified intervals
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("End of video or reading error.")
            break

        # Check if the current frame is at the desired interval
        if frame_count % frame_interval == 0:
            height, width, channels = frame.shape

            # Detect objects using YOLO
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            # Keep track of whether a person has already been detected
            person_detected = False

            # Process detections and draw bounding boxes
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    # Check if the detected object is a person
                    if class_id == classes.index('person') and confidence > 0.2 and not person_detected:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        # Draw bounding box
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, 'person', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        # Set person_detected to True to prevent drawing additional bounding boxes for persons
                        person_detected = True

            # Display the frame only if a person is detected
            if person_detected:
                cv2_imshow(frame)

                # Optional: Add a delay for visualization
                if cv2.waitKey(500) & 0xFF == ord('q'):  # 500 ms delay for observation
                    break

        frame_count += 1  # Increment the frame counter

    # Release the video capture object
    cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

"""PICKLE file for YOLO model"""

# Provide paths to YOLO weights and configuration files
weights_path = "yolov3.weights"
config_path = "yolov3.cfg"

# Load YOLO model
net = cv2.dnn.readNet(weights_path, config_path)

# Get classes from the coco.names file
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Create a dictionary to store YOLO information
yolo_info = {
    "weights_path": weights_path,
    "config_path": config_path,
    "classes": classes,
    "output_layers": output_layers
}

# Specify the full path for saving the pickle file
pickle_file_path = "/content/drive/MyDrive/Yavar/yolo_model.pickle"

# Save YOLO information as a pickle file
with open(pickle_file_path, "wb") as f:
    pickle.dump(yolo_info, f)

print("YOLO model information saved as yolo_model.pickle")

"""YOLO WITH **PREPROCESSING**"""

import cv2
import numpy as np
from google.colab.patches import cv2_imshow
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image
import pickle

def preprocess_frame(frame, target_width=416, target_height=416, normalize=True, denoise=False):
    # Resize frame to specified dimensions
    frame = cv2.resize(frame, (target_width, target_height))

    # Ensure the frame is in BGR (3 channels)
    if frame.shape[2] == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # Ensure the frame is 8-bit unsigned
    if frame.dtype != np.uint8:
        frame = (frame * 255).astype(np.uint8)

    # Apply normalization
    if normalize:
        frame = frame / 255.0

    return frame

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Provide the path to your .mp4 video file
video_path = "/content/drive/MyDrive/Yavar/output_video.mp4"

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video was successfully opened
if not cap.isOpened():
    print("Error: Unable to open the video file.")
else:
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * 1)  # Frame interval
    frame_count = 0  # Frame counter

    # Loop through video frames and preprocess as needed
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if frame_count % frame_interval == 0:
            # Preprocess the frame
            preprocessed_frame = preprocess_frame(frame, normalize=True, denoise=True)

            # Ensure the frame has the correct type and shape
            if preprocessed_frame.dtype == np.uint8 and preprocessed_frame.shape[2] == 3:
                # Detect objects with YOLO
                blob = cv2.dnn.blobFromImage(preprocessed_frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
                net.setInput(blob)
                outs = net.forward(output_layers)

                person_detected = False  # Track if a person is detected

                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]

                        if class_id == classes.index('person') and confidence > 0.2:
                            center_x = int(detection[0] * preprocessed_frame.shape[1])
                            center_y = int(detection[1] * preprocessed_frame.shape[0])
                            w = int(detection[2] * preprocessed_frame.shape[1])
                            h = int(detection[3] * preprocessed_frame.shape[0])
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)

                            # Draw bounding box
                            cv2.rectangle(preprocessed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cv2.putText(preprocessed_frame, 'person', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                            person_detected = True

                if person_detected:
                    cv2_imshow(preprocessed_frame)  # Display the frame with the person detected

        frame_count += 1  # Increment frame count

    cap.release()  # Release resources

# Close all OpenCV windows
cv2.destroyAllWindows()

!pip install mediapipe

"""POSE DETECTION USING MEDIAPIPE"""

import cv2
import mediapipe as mp
from google.colab.patches import cv2_imshow

# Load mediapipe pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Import mediapipe drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Function to detect pose
def detect_pose(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    return results

# Function to draw pose landmarks
def draw_pose_landmarks(frame, results):
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    return frame

# Provide the path to your .mp4 video file
video_path = "/content/drive/MyDrive/Yavar/output_video.mp4"
cap = cv2.VideoCapture(video_path)

# Verify if the video was successfully opened
if not cap.isOpened():
    print("Error: Unable to open the video file.")
else:
    print("Video file opened successfully.")

# Get the FPS of the video
fps = cap.get(cv2.CAP_PROP_FPS)

# Calculate the frame interval to achieve a 0.5-second delay
interval_seconds = 0.5
frame_interval = int(fps * interval_seconds)

# Loop through the video frames and detect pose at the calculated interval
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("No more frames or error reading frame.")
        break

    # Only process frames at the desired interval
    if frame_count % frame_interval == 0:
        # Detect pose
        pose_results = detect_pose(frame)

        # Draw pose landmarks if detected
        if pose_results.pose_landmarks:
            frame_with_pose = draw_pose_landmarks(frame.copy(), pose_results)

            # Display the frame with pose landmarks
            cv2_imshow(frame_with_pose)
            print("Displayed frame with pose landmarks.")

    # Increment the frame count
    frame_count += 1

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting loop.")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

"""PICKLE FILE FOR POSE DETECTION"""

import pickle

# Define the configuration parameters
config_params = {
    "static_image_mode": False,
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5
}

# Define the file path for saving the pickle file
pickle_file_path = "/content/drive/MyDrive/Yavar/pose_config.pickle"

# Save the configuration parameters to a pickle file
with open(pickle_file_path, "wb") as f:
    pickle.dump(config_params, f)

print("Pose configuration parameters saved as pose_config.pickle")

!pip install torch transformers timm mediapipe opencv-python-headless
!pip install transformers
!pip install --upgrade transformers

# Authenticate with Hugging Face
from huggingface_hub import notebook_login
notebook_login()

"""FALL DETECTION USING OWLVIT"""

import cv2
import mediapipe as mp
import numpy as np
from google.colab.patches import cv2_imshow
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image

# Load Mediapipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load OWL-ViT model for open-vocabulary object detection
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

# Function to detect pose
def detect_pose(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    return results

# Function to draw pose landmarks
def draw_pose_landmarks(frame, results):
    if results and results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
        )
    return frame

# Load video
video_path = "/content/drive/MyDrive/Yavar/output_video.mp4"
cap = cv2.VideoCapture(video_path)

# Check if the video was successfully opened
if not cap.isOpened():
    print("Error: Unable to open the video file.")
else:
    print("Video file opened successfully.")

# Get the frames per second (FPS)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Frames per second (FPS): {fps}")

# Calculate the frame interval for processing
interval_seconds = 1
frame_interval = int(fps * interval_seconds)

print(f"Frame interval calculated: {frame_interval}")
frame_count = 0
# Loop through video:
while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        print("No more frames or error reading frame.")
        break

    # Process only frames at the desired interval
    if frame_count % frame_interval == 0:
        # Apply OWL-ViT for object detection
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = processor(text=["person falling"], images=pil_image, return_tensors="pt")
        outputs = model(**inputs)

        # Initialize fall detection flag
        fall_detected = False

        # Extract bounding boxes and logits
        pred_boxes = outputs.pred_boxes[0]  # Get the first set of boxes
        logits = outputs.logits[0]  # Get the first set of logits

        # Set a confidence threshold
        confidence_threshold = -15.4  # Adjust as needed

        array = []

        # Loop through the detected objects
        for i, box in enumerate(pred_boxes):

            logit = logits[i].item()  # Access the corresponding logit
            array.append(logit)

            # Apply confidence threshold
            if logit < confidence_threshold:

                box_coords = box.tolist()  # convert tensor to list

                if len(box_coords) == 4:
                    # Convert box coordinates to pixel values
                    x_min, y_min, x_max, y_max = [
                        int(box_coords[j] * frame.shape[j % 2]) for j in range(4)
                    ]

                    # Draw bounding box if above threshold
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

                    # Indicate fall detection
                    fall_detected = True

        # Detect pose and draw pose landmarks
        pose_results = detect_pose(frame)
        frame_with_pose = draw_pose_landmarks(frame.copy(), pose_results)

        # Display the frame with object detection and pose landmarks
        cv2_imshow(frame_with_pose)

        # Indicate whether a fall was detected
        if fall_detected:
            print("Fall detected in this frame.")
        else:
            print("No fall detected in this frame.")

    # Increment frame count
    frame_count += 1

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

"""PICKLE FILE FOR FALL DETECTION"""

from transformers import OwlViTProcessor, OwlViTForObjectDetection
import pickle

# Initialize the OWL-ViT processor and model
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

# Define the file path for saving the pickle file
pickle_file_path = "/content/drive/MyDrive/Yavar/owlvit_model.pickle"

# Save the processor and model as a tuple in a pickle file
with open(pickle_file_path, "wb") as f:
    pickle.dump((processor, model), f)

print("OWL-ViT model saved as owlvit_model.pickle")

!pip install mediapipe torch torchvision transformers requests

import cv2
import mediapipe as mp
import numpy as np
from google.colab.patches import cv2_imshow
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image

# Load Mediapipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load OWL-ViT model for open-vocabulary object detection
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

# Function to detect pose
def detect_pose(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    return results

# Function to draw pose landmarks
def draw_pose_landmarks(frame, results):
    if results and results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
        )
    return frame

# Load video
video_path = "/content/drive/MyDrive/Yavar/output_video.mp4"
cap = cv2.VideoCapture(video_path)

# Check if the video was successfully opened
if not cap.isOpened():
    print("Error: Unable to open the video file.")
else:
    print("Video file opened successfully.")

# Get the frames per second (FPS)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Frames per second (FPS): {fps}")

# Calculate the frame interval for processing
interval_seconds = 1
frame_interval = int(fps * interval_seconds)

print(f"Frame interval calculated: {frame_interval}")
frame_count = 0
# Loop through video:
while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        print("No more frames or error reading frame.")
        break

    # Process only frames at the desired interval
    if frame_count % frame_interval == 0:
        # Apply OWL-ViT for object detection
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = processor(text=["person falling"], images=pil_image, return_tensors="pt")
        outputs = model(**inputs)

        # Initialize fall detection flag
        fall_detected = False

        # Extract bounding boxes and logits
        pred_boxes = outputs.pred_boxes[0]  # Get the first set of boxes
        logits = outputs.logits[0]  # Get the first set of logits

        # Set a confidence threshold
        confidence_threshold = -15.4  # Adjust as needed

        array = []

        # Loop through the detected objects
        for i, box in enumerate(pred_boxes):

            logit = logits[i].item()  # Access the corresponding logit
            array.append(logit)

            # Apply confidence threshold
            if logit < confidence_threshold:

                box_coords = box.tolist()  # convert tensor to list

                if len(box_coords) == 4:
                    # Convert box coordinates to pixel values
                    x_min, y_min, x_max, y_max = [
                        int(box_coords[j] * frame.shape[j % 2]) for j in range(4)
                    ]

                    # Draw bounding box if above threshold
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

                    # Indicate fall detection
                    fall_detected = True

        # Detect pose and draw pose landmarks
        pose_results = detect_pose(frame)
        frame_with_pose = draw_pose_landmarks(frame.copy(), pose_results)

        # Display the frame with object detection and pose landmarks
        print(min(array))
        cv2_imshow(frame_with_pose)

        # Indicate whether a fall was detected
        if fall_detected:
            print("Fall detected in this frame.")
        else:
            print("No fall detected in this frame.")

    # Increment frame count
    frame_count += 1

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

"""ACCURACY OF THE MODEL"""

import cv2
import mediapipe as mp
import numpy as np
from google.colab.patches import cv2_imshow
from transformers import (
    OwlViTProcessor,
    OwlViTForObjectDetection,
    Owlv2TextConfig,
    Owlv2TextModel,
)
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load OWL-ViT Text configuration and model
text_configuration = Owlv2TextConfig()  # Initializing text configuration
text_model = Owlv2TextModel(text_configuration)  # Initializing text model

# Load Mediapipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load OWL-ViT model for open-vocabulary object detection
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

# Function to detect pose
def detect_pose(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    return results

# Function to draw pose landmarks
def draw_pose_landmarks(frame, results):
    if results and results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
        )
    return frame

# Define ground truth for fall detection (example)
# 0 means no fall, 1 means fall; replace with your actual ground truth data
ground_truth = [0, 0, 1, 0, 0, 1, 0, 0, 0]  # Modify based on your video content

# Store predictions for accuracy calculation
predictions = []

# Load video
video_path = "/content/drive/MyDrive/Yavar/output_video.mp4"  # Update with your video path
cap = cv2.VideoCapture(video_path)

# Check if the video was successfully opened
if not cap.isOpened():
    print("Error: Unable to open the video file.")
else:
    print("Video file opened successfully.")

# Get the frames per second (FPS)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps * 1)  # Frame interval, adjust as needed
frame_count = 0  # Frame counter

# Loop through video frames and process at the desired interval
while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        print("No more frames or error reading frame.")
        break

    if frame_count % frame_interval == 0:
        # Apply OWL-ViT for object detection
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = processor(text=["person falling"], images=pil_image, return_tensors="pt")
        outputs = model(**inputs)

        fall_detected = False  # Flag for fall detection

        # Extract bounding boxes and logits
        pred_boxes = outputs.pred_boxes[0]  # Get the first set of boxes
        logits = outputs.logits[0]  # Get the first set of logits

        confidence_threshold = -15.4  # Set a confidence threshold
        logit_array = []  # To store logit values for debugging

        for i, box in enumerate(pred_boxes):
            logit = logits[i].item()
            logit_array.append(logit)  # Store logit for debugging

            # Apply confidence threshold
            if logit < confidence_threshold:
                box_coords = box.tolist()  # Convert to list
                if len(box_coords) == 4:
                    x_min, y_min, x_max, y_max = [
                        int(box_coords[j] * frame.shape[j % 2]) for j in range(4)
                    ]
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                    fall_detected = True  # Flag fall detection

        # Detect pose and draw pose landmarks
        pose_results = detect_pose(frame)
        frame_with_pose = draw_pose_landmarks(frame.copy(), pose_results)

        # Display frame with object detection
        cv2_imshow(frame_with_pose)

        # Store whether a fall was detected for this frame
        predictions.append(1 if fall_detected else 0)

    # Increment frame count
    frame_count += 1

    # Break the loop if the number of predictions matches the length of ground_truth
    if len(predictions) >= len(ground_truth):
        break

# Ensure the length of predictions matches the length of ground truth labels
while len(predictions) < len(ground_truth):
    predictions.append(0)  # Assume no fall if no prediction is made for a frame

# Release resources
cap.release()
cv2.destroyAllWindows()

# Calculate precision score
if sum(predictions) == 0 and sum(ground_truth) == 0:
    precision = 1.0  # Set precision to 1.0 if no positive predictions and no positive instances in ground truth
else:
    precision = precision_score(ground_truth, predictions, zero_division=0)  # Use zero_division parameter to handle division by zero

# Calculate accuracy metrics
accuracy = accuracy_score(ground_truth, predictions)
precision = precision_score(ground_truth, predictions)
recall = recall_score(ground_truth, predictions)
f1 = f1_score(ground_truth, predictions)
# Calculate true negatives (TN)
true_negatives = sum([1 for gt, pred in zip(ground_truth, predictions) if gt == 0 and pred == 0])

# Calculate false positives (FP)
false_positives = sum([1 for gt, pred in zip(ground_truth, predictions) if gt == 0 and pred == 1])

# Calculate false positive rate (FPR)
if false_positives + true_negatives == 0:
    fpr = 0  # Handle division by zero
else:
    fpr = false_positives / (false_positives + true_negatives)

print(f"False Positive Rate (FPR): {fpr:.2f}")

print(f"Accuracy: {accuracy:.2f}")