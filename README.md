SpeakVision is a deep learning-based application designed for lip-reading, leveraging the powerful LipNet model. This project utilizes computer vision and speech-to-text techniques to transcribe spoken words by analyzing video footage of lip movements. The application takes video input and provides real-time transcription of the speech in the video.

Features
-Video Input: Users can upload or select video files containing speech.
-Machine Learning Model: The core of the application is based on LipNet, a state-of-the-art lip-reading model that transcribes speech from video frames.
-Real-time Transcription: The app processes the video frames, decodes lip movements, and outputs the transcription as text.
=Visualization: Displays both the video and the machine-readable representation (grayscale video frames) for a better understanding of the model's process.

Technologies Used
-TensorFlow: For building and loading the machine learning model.
-OpenCV: For video processing and frame extraction.
-Streamlit: For building the interactive web application.
-imageio: For creating GIFs to visualize processed video frames.
