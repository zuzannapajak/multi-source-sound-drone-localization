import h5py

filepath = "C:/Users/pajak/OneDrive - Politechnika Łódzka/Dokumenty/Pulpit/Multi-Source-Sound-Localization/data/Drones_data.h5"

with h5py.File(filepath, 'r') as f:
    print("Audio shape:", f['audio'].shape)  # Check the shape of the audio dataset
    print("Images shape:", f['images'].shape)  # Check the shape of the images dataset
    print("Sample 0 audio:", f['audio'][0])  # Print audio data at index 0
    print("Sample 0 images:", f['images'][0])  # Print images data at index 0
    
    print(f.keys())  # See available datasets inside the HDF5 file
    print(f['audio'].shape)  # Check the shape of the audio dataset
    print(f['images'].shape)  # Check the shape of the images dataset, since 'video' doesn't exist

    # Inspect data at index 0
    audio_data = f['audio'][0]
    images_data = f['images'][0]  # Use 'images' instead of 'video'
    
    print("Audio data:", audio_data)
    print("Images data:", images_data)  # Print the image data
