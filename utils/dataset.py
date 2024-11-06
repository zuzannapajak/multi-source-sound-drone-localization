import torch
import torch.utils.data as data
import os
import numpy as np
import random
import h5py
import librosa

EPS = np.spacing(1)

def DataAllocate(batch):
    audios = []
    visuals = []
    rois = []

    for sample in batch:
        if sample is None:
            continue  # Skip None values

        audios.append(sample[0])
        images.append(sample[1])

        if len(sample) > 2:  # Check if 'rois' exists
            rois.append(sample[2])

    # Handle missing rois gracefully
    if len(rois) == 0:  # In case rois are not part of the data
        rois = None

    audios = torch.stack(audios, dim=0)  # (batchsize, mix, T, F)
    images = torch.stack(images, dim=0).permute(0, 1, 2, 5, 3, 4).contiguous()  # (batchsize, mix, 9, 256, 256, 3)

    if rois:
        rois = torch.stack(rois, dim=0)  # (batchsize, mix, 9, 8, 4)
    
    return audios, images, rois
    
class AudioVisualData(data.Dataset):
    
    def __init__(self, audio, images, rois, mix, frame, dataset, training):
        self.sr = 22050
        self.mix = mix
        self.frame = frame
        self.dataset = dataset
        self.training = training
        self.base = 3339 if self.dataset == 'AVE_C' else 0
        if self.training:
            self.samples = 3339 if self.dataset == 'AVE_C' else 10000
        else:
            self.samples = 402 if self.dataset == 'AVE_C' else 500

        self.audio = h5py.File(audio, 'r')
        self.images = h5py.File(images, 'r') 

        if rois:
            self.rois = h5py.File(rois, 'r')
        else:
            self.rois = None

    def __len__(self):
        return self.samples
    
    def get_train(self, idx):
    # If there's only one sample, use it instead of trying to sample multiple
        if self.samples == 1:
            train_idx = [0]  # Use the only sample available
        else:
            train_idx = np.random.permutation(self.samples)[:self.mix]        
        audio = []
        images = []
        rois = []
        spec = []

        for i in range(self.mix):
            spec_i = None
            rois_i = None
            images_i = None

            if self.dataset == 'AVE_C':  # AVE
                start = np.random.randint(0, 10 - self.frame)
                try:
                    # Repeat audio if fewer than image samples
                    audio_idx = i % num_audio  # Loop over the available audio data
                    spec_i = self.audio['audio'][audio_idx, start: start + self.frame].reshape(-1, 64)
                    rois_i = self.rois['roi'][train_idx[i], start: start + self.frame] * 256
                    images_i = (self.images['images'][train_idx[i], start: start + self.frame] / 255.0 - 
                                np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
                except Exception as e:
                        print(f"Skipping sample {train_idx[i]} due to missing data or error: {e}")
                        continue

            # Add valid data
            if spec_i is not None and rois_i is not None and images_i is not None:
                spec.append(spec_i)
                rois.append(rois_i)
                images.append(images_i)

        if len(spec) == 0 or len(rois) == 0 or len(images) == 0:
            print(f"Skipping index {idx} due to empty data.")
            return None

        # Convert to torch tensors
        spec = torch.FloatTensor(spec)
        rois = torch.FloatTensor(rois)
        images = torch.FloatTensor(images)

        return spec, images, rois

    
    def get_val(self, idx):
        val_idx = np.random.permutation(self.samples-1)
        val_idx[val_idx>=idx] += 1
        val_idx = np.hstack([np.array(idx), val_idx])
        val_idx = val_idx + self.base
        audio = [None] * self.mix
        images = [None] * self.mix
        rois = [None] * self.mix
        spec = [None] * self.mix
        for i in range(self.mix):
            if self.dataset == 'AVE_C':#AVE
                start = np.random.randint(0, 10-self.frame)
                spec[i] = self.audio['audio'][val_idx[i], start: start+self.frame].reshape(-1, 64)
                rois[i] = self.rois['roi'][val_idx[i], start: start+self.frame] * 256
                images[i] = (self.images['images'][val_idx[i], start: start+self.frame]/255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            if self.dataset == 'Flickr':#Flickr
                spec[i] = self.audio['audio'][val_idx[i]]
                rois[i] = self.rois['roi'][val_idx[i]] * 256
                rois[i] = np.expand_dims(rois[i], 0)
                images[i] = (self.images['images'][val_idx[i]]/255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
                images[i] = np.expand_dims(images[i], 0)

        spec = torch.FloatTensor(spec)
        rois = torch.FloatTensor(rois)
        images = torch.FloatTensor(images)
        
        return spec, images, rois
    
    def __getitem__(self, idx):
        if idx >= len(self.audio['audio']) or idx >= len(self.images['images']):
            print(f"Skipping index {idx} because it exceeds dataset length.")
            return None

        for _ in range(10):  # Retry 10 times before giving up
            if self.training:
                result = self.get_train(idx)
            else:
                result = self.get_val(idx)

            if result is not None:
                return result
                
            print(f"Skipping index {idx} due to empty data or errors. Retrying...")

        # If after retries, we still get None, raise an error or handle as necessary
        raise ValueError(f"All retries failed for index {idx}.")

