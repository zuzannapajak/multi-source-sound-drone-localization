import torch
import torch.utils.data as data
import os
import numpy as np
import pickle
import cv2
import h5py
import librosa

EPS = np.spacing(1)


def DataAllocate(batch):
    audios = []
    visuals = []
    labels = []
    for sample in batch:
        if sample is None:
            continue  # Skip None values

        audios.append(sample[0])
        visuals.append(sample[1])
        labels.append(sample[2])

    # Return None if the batch has no valid samples
    if len(audios) == 0 or len(visuals) == 0 or len(labels) == 0:
        return None

    audios = torch.stack(audios, dim=0)
    visuals = torch.stack(visuals, dim=0)
    labels = torch.stack(labels, dim=0)

    return audios, visuals, labels


class AudioVisualData(data.Dataset):

    def __init__(self, hdf5_file, mix, frame, training):
        self.sr = 22050  # Sample rate
        self.mix = mix  # Number of samples to mix together
        self.frame = frame  # Number of frames to use
        self.training = training  # Flag for training or validation mode
        self.mean = np.array([0.485, 0.456, 0.406])  # Image normalization mean
        self.std = np.array([0.229, 0.224, 0.225])  # Image normalization std

        # Load the HDF5 file
        self.h5_file = h5py.File(hdf5_file, 'r')
        self.audio_data = self.h5_file['audio']
        self.image_data = self.h5_file['images']

        self.samples = len(self.audio_data)

    def __len__(self):
        return self.samples

    def get_train(self):
        specs = []
        visuals = []
        labels = []

        audio_len = self.frame * self.sr
        visual = np.zeros((self.frame, 256, 256, 3))

        # Randomly choose self.mix samples to mix
        train_idx = np.random.permutation(self.samples)[:self.mix]

        for idx in train_idx:
            start_frame = np.random.randint(0, 10 - self.frame)

            # Load images from HDF5 file
            visual_data = self.image_data[idx, start_frame: start_frame + self.frame]
            visual_data = (visual_data / 255.0 - self.mean) / self.std
            visuals.append(visual_data)

            # Load audio and generate spectrogram
            audio = self.audio_data[idx, start_frame: start_frame + self.frame]
            if len(audio) < audio_len:
                padded_audio = np.zeros(audio_len)
                padded_audio[:len(audio)] = audio
                audio = padded_audio

            spec = librosa.feature.melspectrogram(audio[:audio_len], self.sr, n_fft=882, hop_length=441, n_mels=64)
            spec = np.log(spec + EPS).T
            specs.append(spec)

            # Placeholder for labels, modify based on your actual labeling mechanism
            labels.append(np.zeros(1))  # Modify this based on your label format

        return specs, visuals, labels

    def get_val(self, idx):
        specs = []
        visuals = []
        labels = []

        audio_len = self.frame * self.sr
        visual = np.zeros((self.frame, 256, 256, 3))

        start_frame = np.random.randint(0, 10 - self.frame)

        # Load validation images
        visual_data = self.image_data[idx, start_frame: start_frame + self.frame]
        visual_data = (visual_data / 255.0 - self.mean) / self.std
        visuals.append(visual_data)

        # Load validation audio and generate spectrogram
        audio = self.audio_data[idx, start_frame: start_frame + self.frame]
        if len(audio) < audio_len:
            padded_audio = np.zeros(audio_len)
            padded_audio[:len(audio)] = audio
            audio = padded_audio

        spec = librosa.feature.melspectrogram(audio[:audio_len], self.sr, n_fft=882, hop_length=441, n_mels=64)
        spec = np.log(spec + EPS).T
        specs.append(spec)

        # Placeholder for labels
        labels.append(np.zeros(1))  # Modify based on your label format

        return specs, visuals, labels

    def __getitem__(self, idx):
        # For training, get the training data
        if self.training:
            specs, visuals, labels = self.get_train()
        else:
            # For validation, get the validation data
            specs, visuals, labels = self.get_val(idx)

        specs = torch.FloatTensor(specs)
        visuals = torch.FloatTensor(visuals).permute(0, 4, 1, 2).contiguous()  # Adjust dimensions to PyTorch format
        labels = torch.FloatTensor(labels)

        return specs, visuals, labels

