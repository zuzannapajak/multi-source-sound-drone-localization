import librosa
import numpy as np
import os
import matplotlib.pyplot as plt

def convert_audio_to_melspectrogram(audio_path, output_dir, sr=22050, n_mels=128):
    # Get the base name of the audio file (e.g., Drones_1 or Drones_2)
    audio_name = os.path.splitext(os.path.basename(audio_path))[0]
    
    # Create the output directory if it doesn’t exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the output path for the spectrogram
    output_path = os.path.join(output_dir, f"{audio_name}.npy")
    
    # Load the audio file and compute the mel spectrogram
    y, sr = librosa.load(audio_path, sr=sr)
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    
    # Save the mel spectrogram as a numpy array
    np.save(output_path, log_mel_spect)
    
    # Optionally, visualize the mel spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(log_mel_spect, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Mel spectrogram for {audio_name}')
    plt.tight_layout()
    plt.show()

# Example usage:
convert_audio_to_melspectrogram("audio/Drones_1.wav", "audio/spectrograms")
convert_audio_to_melspectrogram("audio/Drones_2.wav", "audio/spectrograms")
