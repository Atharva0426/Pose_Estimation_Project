import streamlit as st
from PIL import Image
import numpy as np
import cv2

DEMO_IMAGE = 'stand.jpg'
BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

width = 368
height = 368
inWidth = width
inHeight = height

net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")

# Page layout and title
st.set_page_config(page_title="Pose Estimation App", layout="wide")
st.title("📷 Human Pose Estimation App")
st.markdown("---")

# Sidebar for input options
st.sidebar.header("About")
st.sidebar.markdown("An AI-powered pose estimation app designed to detect and map human body keypoints from images, videos, or live camera feeds.")
st.sidebar.header("Input Options")
st.sidebar.markdown("Select the input type for pose estimation.")
input_type = st.sidebar.radio("Choose input type:", ("Image", "Video", "Camera"))

st.sidebar.markdown("---")
thres = st.sidebar.slider('Keypoint Detection Threshold', min_value=0, value=20, max_value=100, step=5) / 100
st.sidebar.markdown("Adjust the slider to set the confidence threshold for detecting keypoints.")

# Helper function
@st.cache_data
def pose_detector(frame):
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]

    assert(len(BODY_PARTS) == out.shape[1])
    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > thres else None)

    for pair in POSE_PAIRS:
        partFrom, partTo = pair
        idFrom, idTo = BODY_PARTS[partFrom], BODY_PARTS[partTo]
        if points[idFrom] and points[idTo]:
            cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

    return frame

# Main content section
if input_type == "Image":
    st.header("📸 Upload an Image")
    img_file_buffer = st.file_uploader("Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"], help="Ensure the image has visible body parts for better pose estimation.")

    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))
        st.image(image, caption="Uploaded Image", use_container_width=True)

        output = pose_detector(image)
        st.subheader("Pose Estimation Result")
        st.image(output, caption="Pose Estimated", use_column_width=True)
    else:
        st.warning("Please upload an image to proceed.")

elif input_type == "Video":
    st.header("🎥 Upload a Video")
    video_file_buffer = st.file_uploader("Upload a video (MP4, AVI, MOV)", type=["mp4", "avi", "mov"], help="Ensure the video has visible body parts for pose estimation.")

    if video_file_buffer is not None:
        video_path = video_file_buffer.name
        tfile = open(video_path, "wb")
        tfile.write(video_file_buffer.read())
        cap = cv2.VideoCapture(video_path)

        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            output = pose_detector(frame)
            stframe.image(output, channels="BGR", use_column_width=True)
        cap.release()
    else:
        st.warning("Please upload a video to proceed.")

elif input_type == "Camera":
    st.header("📹 Real-Time Camera Feed")
    st.text("Using the system camera for real-time pose estimation.")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Unable to access the camera.")
    else:
        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            output = pose_detector(frame)
            stframe.image(output, channels="BGR", use_column_width=True)
        cap.release()

st.markdown("---")
st.footer = ("Developed using OpenCV and Streamlit. For best results, use clear images or videos with visible body parts.")
