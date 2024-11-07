import h5py
import numpy as np
import os
import cv2

def create_hdf5(image_root_dir, audio_root_dir, output_path):
    # Create the HDF5 file
    with h5py.File(output_path, 'w') as f:
        # Iterate over each video directory within the frames directory
        for video_name in sorted(os.listdir(image_root_dir)):
            video_image_dir = os.path.join(image_root_dir, video_name)
            video_audio_path = os.path.join(audio_root_dir, f"{video_name}.npy")  # Corresponding audio spectrogram path
            
            if os.path.isdir(video_image_dir) and os.path.exists(video_audio_path):
                images = []
                # Load all frames for this video
                for img in sorted(os.listdir(video_image_dir)):
                    if img.endswith(".jpg"):
                        image_path = os.path.join(video_image_dir, img)
                        img_data = cv2.imread(image_path)
                        img_data = cv2.resize(img_data, (256, 256))  # Resize as needed
                        images.append(img_data)
                
                # Create a group for each video and store frames
                video_group = f.create_group(video_name)
                video_group.create_dataset('images', data=np.array(images))
                
                # Load and add the corresponding audio spectrogram for this video
                spectrogram = np.load(video_audio_path)
                video_group.create_dataset('audio', data=spectrogram)
        
    print(f"HDF5 file saved at {output_path}")

# Example usage:
image_root_directory = 'images/frames'  # Root directory containing subdirectories of frames for each video
audio_root_directory = 'audio/spectrograms'  # Directory containing spectrograms for each video
output_hdf5_path = 'Drones_data.h5'  # Output path for HDF5 file

create_hdf5(image_root_directory, audio_root_directory, output_hdf5_path)
