# Import all of the dependencies
import streamlit as st
import os 
import imageio 
import numpy as np
import tensorflow as tf 
from utils import load_data, num_to_char
from modelutil import load_model

# Set the layout to the streamlit app as wide 
st.set_page_config(layout='wide')

# Setup the sidebar
with st.sidebar: 
    st.image(r"..\logo.jpeg")
    st.title('Visual Speech Recognition')
    st.info('End-to-End Sentence Level LipReading.')

st.title('LipNet App') 
# Generating a list of options or videos 
options = os.listdir(os.path.join('..', 'data', 's1'))
selected_video = st.selectbox('Choose video', options)

# Generate two columns 
col1, col2 = st.columns(2)

if options: 

    # Rendering the video 
    with col1: 
        st.info('The video below displays the converted video in mp4 format')
        
        # Get the absolute path of the selected video
        file_path = os.path.join('..', 'data', 's1', selected_video)
        
        # Dynamically generate the output file name based on the selected video
        output_file = os.path.join('output_videos', f"{os.path.splitext(selected_video)[0]}_converted.mp4")
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Using FFmpeg to convert the selected video to mp4 format
        os.system(f'ffmpeg -i "{file_path}" -vcodec libx264 "{output_file}" -y')

        # Rendering the converted video inside of the app
        with open(output_file, 'rb') as video:
            video_bytes = video.read() 
            st.video(video_bytes)




    with col2: 
        st.info('This is all the machine learning model sees when making a prediction')
        video, annotations = load_data(tf.convert_to_tensor(file_path))

        # Convert the video tensor to a NumPy array and ensure its data type is uint8
        video_np = video.numpy()
        video_np = (video_np * 255).astype(np.uint8)

        # Ensuring the shape of the video is compatible
        video_np = video_np.squeeze()

        imageio.mimsave('animation.gif', video_np, fps=10)
        st.image('animation.gif', width=400) 

        st.info('This is the output of the machine learning model as tokens')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        # Convert prediction to text
        st.info('Decode the raw tokens into words')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)
