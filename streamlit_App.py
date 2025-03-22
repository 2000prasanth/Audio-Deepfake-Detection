import numpy as np
import librosa
import tensorflow as tf
import streamlit as st

window_length = 0.02  # 20ms window length
hop_length = 0.0025  # 2.5ms hop length
sample_rate = 22050  # Standard audio sample rate
n_mels = 128  # Number of mel filter banks
threshold_zcr = 0.1  # Adjust this threshold to detect breath based on ZCR
threshold_rmse = 0.1  # Adjust this threshold to detect breath based on RMSE

def extract_breath_features(y, sr):
    frame_length = int(window_length * sr)
    hop_length_samples = int(hop_length * sr)
    
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=frame_length, hop_length=hop_length_samples)
    rmse = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length_samples)
    
    zcr = zcr.T.flatten()
    rmse = rmse.T.flatten()
    
    # Calculate breath events
    breaths = (zcr > threshold_zcr) & (rmse > threshold_rmse)
    
    # Create a breath feature: 1 if breath is present, else 0
    breath_feature = np.where(breaths, 1, 0)
    
    return breath_feature

def extract_features(file_path, n_mels=128, n_cqt=84, max_len=500, n_mfcc=13):
    try:
        y, sr = librosa.load(file_path, sr=None)
        
        # Compute MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc = librosa.util.fix_length(mfcc, size=max_len, axis=1)  # Fix length
        
        # Compute log-mel spectrogram
        logspec = librosa.amplitude_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels))
        logspec = librosa.util.fix_length(logspec, size=max_len, axis=1)  # Fix length
        
        # Extract breath features
        breath_feature = extract_breath_features(y, sr)
        breath_feature = librosa.util.fix_length(breath_feature, size=max_len)  # Fix length
        
        # Stack features vertically
        return np.vstack((mfcc,logspec, breath_feature))
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Function to prepare the features for prediction
def prepare_single_data(features, max_len=500):
    features = librosa.util.fix_length(features, size=max_len, axis=1)
    features = features[np.newaxis, ..., np.newaxis]  # Add batch and channel dimensions
    return features

# Load the saved TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path=r"model_breath_logspec_mfcc_cnn.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to predict audio class
def predict_audio(file_path):
    features = extract_features(file_path)
    if features is not None:
        prepared_features = prepare_single_data(features)
        # Ensure the prepared features are of type FLOAT32
        prepared_features = prepared_features.astype(np.float32)  # Convert to FLOAT32
        # Set the tensor to the prepared input data
        interpreter.set_tensor(input_details[0]['index'], prepared_features)
        interpreter.invoke()
        # Get the prediction result
        prediction = interpreter.get_tensor(output_details[0]['index'])
        predicted_class = np.argmax(prediction, axis=1)
        predicted_prob = prediction[0]  # Get the probabilities for EER calculation
        return predicted_class[0], predicted_prob  # Return class index and probabilities
    else:
        return None, None

# Streamlit app
st.title('Audio Classification: Real vs Fake')
st.write('Upload an audio file to classify it as real or fake.')

# File uploader
uploaded_file = st.file_uploader('Choose an audio file', type=['wav', 'mp3'])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open('temp_audio_file.wav', 'wb') as f:
        f.write(uploaded_file.getbuffer())


    # Predict using the loaded model
    prediction,probablity = predict_audio('temp_audio_file.wav')
    st.write(f'Predicted class is {prediction} \n')
    st.write(f'Probability of being real: {probablity[0]*100:.2f}% \n')
    st.write(f'Probability of being fake: {probablity[1]*100:.2f}% \n')
    

    
    
    

 
