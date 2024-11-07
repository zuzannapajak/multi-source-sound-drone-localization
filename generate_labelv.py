import numpy as np
from resnet.resnet import resnet18
import torch
import torch.nn as nn
import h5py
import time

if __name__ == "__main__":
    labels = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load ResNet with modified input channels
    model = resnet18(pretrained=False, num_classes=1000)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Adjust input channels to 3
    model.to(device)
    model.eval()

    spectrogram_files = [
        'data/audio/spectrograms/Drones_1.npy',
        'data/audio/spectrograms/Drones_2.npy'
    ]

    images_file_path = 'data/Drones_data.h5'
    images = h5py.File(images_file_path, 'r')

    for spec_file in spectrogram_files:
        aggre = np.load(spec_file)  

        with torch.no_grad():
            for im_data in images['Drones_1/images']:  
                start_time = time.time()
                
                # Normalize image data and reshape to the required 4D tensor
                im = (im_data / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
                im = torch.FloatTensor(im).permute(2, 0, 1).unsqueeze(0).to(device)  # Move to GPU if available

                # Expand `im` to 256 channels by repeating
                im = im.repeat(1, 85, 1, 1)  # Repeat channels to 255
                im = torch.cat((im, im[:, :1, :, :]), dim=1)  # Concatenate one more channel to reach 256

                # Forward pass through the model
                output = model(im, mode="eval")
                if isinstance(output, tuple):
                    prob = output[0]
                else:
                    prob = output

                # Detach and convert `prob` to numpy for manipulation
                prob = prob.detach().cpu().numpy()  # Move to CPU if needed

                # Adjust `prob` to match the shape of `aggre`
                prob_expanded = np.repeat(prob, 128, axis=0)  # Expands prob along the first dimension to match aggre

                # Perform element-wise multiplication
                if prob_expanded.shape[1] > aggre.shape[1]:
                    prediction = prob_expanded[:, :aggre.shape[1]] * aggre
                else:
                    prediction = prob_expanded * aggre[:, :prob_expanded.shape[1]]

                # Normalize prediction
                prediction = np.sum(prediction, axis=1)
                prediction = prediction / np.max(prediction)
                labels.append(prediction)

                end_time = time.time()
                print(f"Processed one image in {end_time - start_time:.2f} seconds")

    np.save('labels_v', labels)
