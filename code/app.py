import streamlit as st
import base64
from PIL import Image
import pickle
import cv2
import numpy as np
import mediapipe as mp


# Define the Streamlit app layout
st.set_page_config(
    layout="wide",
    page_title="FALL DETECTION",
    page_icon="🌸",  # You can change this to your preferred icon
)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def detect_person(frame, yolo_weights, yolo_cfg, coco_names, pickle_file):

    frame = frame.astype(np.uint8)    
    st.image(frame, width=900)
    # Load YOLO model and configuration files
    net = cv2.dnn.readNet(yolo_weights, yolo_cfg)

    # Function to resize frames
    def resize_frames(frames, target_width=640, target_height=480):
        resized_frames = [cv2.resize(frame, (target_width, target_height)) for frame in frames]
        return resized_frames

    classes = []
    with open(coco_names, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # Load predictions from the pickle file
    predictions = load_predictions_from_pickle(pickle_file)

    # Check if the frame was read successfully
    if frame is None:
        print("Error: Input frame is None.")
        return

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

    # Return True if a person is detected, False otherwise
    return person_detected

def load_predictions_from_pickle(pickle_file):
    with open(pickle_file, 'rb') as f:
        predictions = pickle.load(f)
    return predictions

def estimate_pose(detected_person):
    # Convert the detected person region to grayscale
    frame_rgb = cv2.cvtColor(detected_person, cv2.COLOR_BGR2RGB)
    
    # Perform pose detection using MediaPipe Pose model
    results = pose.process(frame_rgb)
    
    # Draw pose landmarks on the detected person image
    frame_with_pose_landmarks = draw_pose_landmarks(detected_person, results)

    return frame_with_pose_landmarks

# Function to draw pose landmarks
def draw_pose_landmarks(frame, results):
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))
    return frame

def main():
    st.title('Fall Detection App')

    tab1, tab2, tab3 = st.tabs(["FALL DETECTION", "PREDICTION YOLO MODEL", "POSE MODEL PREDICTION"])

    with tab1:
        st.header("Fall Images")
        # Path to the two fall detection images
        image1_path = 'images/fall1.png'
        image2_path = 'images/fall2.png'

        # Display the first image
        st.image(image1_path, caption='FALL DETECTION 1', width=900)

        # Display the second image
        st.image(image2_path, caption='FALL DETECTION 2', width=900)

    with tab2:
        st.header("Predictions using YOLO for a single frame in the video")

        # Upload image file
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Read image from uploaded file
            image = Image.open(uploaded_file)

            # Convert image to OpenCV format
            frame = np.array(image)
            
            # Define paths to YOLO files
            yolo_weights = "yolov3.weights"
            yolo_cfg = "yolov3.cfg"
            coco_names = "coco.names"
            pickle_file = 'yolo_model.pickle'

            # Perform fall detection
            person_detected = detect_person(frame, yolo_weights, yolo_cfg, coco_names, pickle_file)

            # Display result
            if person_detected:
                st.write("# Person detected!",font_size=50)
            else:
                st.write("# Not Person detected!",font_size=50)
                
    with tab3:
        st.header("Prediction using Pose estimation model for a single frame in the video")
        uploader_key = "image_uploader"
        # File uploader for image
        uploaded_files = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"],key=uploader_key)

        if uploaded_files is not None:
            # Read the uploaded image
            image = cv2.imdecode(np.fromstring(uploaded_files.read(), np.uint8), 1)
            
            # Load YOLO model for person detection
            yolo_weights = "yolov3.weights"
            yolo_cfg = "yolov3.cfg"
            coco_names = "coco.names"
            net = cv2.dnn.readNet(yolo_weights, yolo_cfg)

            pickle_file='pose_model.pickle'
            pickle_file1 = 'yolo_model.pickle'
            
            # Perform person detection
            detected_person = detect_person(image,yolo_weights,yolo_cfg, coco_names,pickle_file1)

            if detected_person:
                # Perform pose estimation
                skeleton_image = estimate_pose(image)

                # Display original image with skeleton overlay
                st.write("# POSE ESTIMATION",font_size=50)
                st.image(skeleton_image, caption='Pose Estimation', width=900)
            else:
                st.write("# No person detected in the image.")
    

if __name__ == "__main__":
    main()
