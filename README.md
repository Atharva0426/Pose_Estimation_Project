# Pose Estimation System

## Overview
This project is a human pose estimation system designed to detect and map key body points (joints and limbs) from images, videos, and live camera feeds. By leveraging deep learning and pre-trained models, it provides an accurate, scalable, and efficient solution for real-world applications such as healthcare, sports 
## Technologies Used
- **OpenCV**: For image and video processing, as well as loading the pre-trained model.
- **TensorFlow**: For the deep learning model (`graph_opt.pb`) used for pose estimation.
- **Streamlit**: For creating an intuitive user interface.
- **NumPy**: For numerical computations and data handling.
- **Pillow (PIL)**: For image loading and manipulation.

## How It Works
1. **Input Handling**: Users upload an image, video, or use a live camera feed.
2. **Preprocessing**: Inputs are resized and normalized to match the model’s requirements.
3. **Pose Estimation**: The pre-trained model (`graph_opt.pb`) detects keypoints and connects them using Part Affinity Fields (PAFs).
4. **Visualization**: Keypoints and skeletons are drawn on the input image or video.
5. **Output Display**: The processed output is displayed through the Streamlit interface.

   ## Acknowledgments
This project was developed as part of the AICTE Internship on AI: Transformative Learning, supported by TechSaksham, a joint CSR initiative by Microsoft and SAP.
