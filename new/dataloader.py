import torch
import torch.utils.data as data
import numpy as np
import h5py
import os

def DataAllocate(batch):   
    specs = []
    imgs = []
    labels_a = []
    labels_v = []
    for sample in batch:
        specs.append(sample[0])
        imgs.append(sample[1])
        labels_a.append(sample[2])
        labels_v.append(sample[3])
    specs = torch.stack(specs, 0).unsqueeze(1)
    imgs = torch.stack(imgs, 0).permute(0, 3, 1, 2).contiguous()
    labels_a = torch.stack(labels_a, 0)
    labels_v = torch.stack(labels_v, 0)
    return specs, imgs, labels_a, labels_v


class AudioVisualData(data.Dataset):
    
    def __init__(self, data_path):
        # Open the HDF5 file
        self.data_path = data_path
        self.h5_file = h5py.File(self.data_path, 'r')
        
        # Access the datasets within the HDF5 file
        self.audio = self.h5_file['Drones_1/audio']  
        self.video = self.h5_file['Drones_1/video']
        self.audio = self.h5_file['Drones_2/audio']  
        self.video = self.h5_file['Drones_2/video']
        
        # You may need to load labels if they exist in the file or define dummy labels for testing
        if 'labels_a' in self.h5_file and 'labels_v' in self.h5_file:
            self.label_a = self.h5_file['labels_a'][:]
            self.label_v = self.h5_file['labels_v'][:]
        else:
            # Dummy labels if they are not stored in the HDF5 file
            self.label_a = np.zeros((self.audio.shape[0], 10))  # Assuming 10 classes for example
            self.label_v = np.zeros((self.video.shape[0], 10))  # Adjust the dimensions as needed

    def __len__(self):
        return len(self.audio)  # Ensure it matches the number of samples

    def __getitem__(self, idx):
        spec = self.audio[idx]
        img = self.video[idx]
        label_a = self.label_a[idx]
        label_v = self.label_v[idx]
        
        # Optional: Random horizontal flip for data augmentation
        if np.random.rand() > 0.5:
            img = img[:, ::-1]
        
        # Normalize the image data
        img = (img / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        
        # Return tensors
        return torch.FloatTensor(spec), torch.FloatTensor(img), torch.FloatTensor(label_a), torch.FloatTensor(label_v)

    def __del__(self):
        # Close the HDF5 file when done
        self.h5_file.close()
