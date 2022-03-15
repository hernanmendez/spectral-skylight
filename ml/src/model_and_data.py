import torch
import torch.nn as nn
import pandas as pd

class SkylightData():
    spectrum_beginning = 350
    spectrum_end = 1780
    wavelengths = [wavelength for wavelength in range(spectrum_beginning, spectrum_end + 1)]
    
    def __init__(self, skies_csv_filepath):
        self.sky_data = pd.read_csv(skies_csv_filepath)
        
        measurements = self.sky_data[map(str, self.wavelengths)]

        self.data_mean = torch.tensor(measurements.mean(0))
        self.data_range = torch.tensor(measurements.max(0) - measurements.min(0))
        
    def normalize(self, raw_spectral_randiance_tensor):
        return (raw_spectral_randiance_tensor - self.data_mean) / self.data_range
    
    def denormalize(self, normalized_randiance_tensor):
        return (normalized_randiance_tensor * self.data_range) + self.data_mean
        
class SkylightModel():
    @classmethod
    def create_model(cls, device):
        return nn.Sequential(
            nn.Conv2d(3 * 8, 64, 7, padding=(3, 3)), # 100x100
            nn.MaxPool2d(2, 2), # 50x50
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 5, padding=(2, 2)), # 50x50
            nn.MaxPool2d(2, 2), # 25, 25
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, 3, padding=(1, 1)), # 25x25
            nn.AdaptiveMaxPool2d(16), # 16x16
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, 2, stride=(2, 2), padding=(0, 0)), # 8x8
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 1024, 2, stride=(2, 2), padding=(0, 0)), # 4x4
            nn.ReLU(inplace=True),

            nn.Flatten(),

            nn.Dropout(),
            nn.Linear(1024 * 4 * 4, 8192),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.7),
            nn.Linear(8192, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, len(SkylightData.wavelengths))
        ).to(device)
        
    
    def __init__(
            self,
            training_data = SkylightData("..\..\..\coords-scattered-hdr.csv"),
            device='cuda',
            save_location = '..\..\..\sky_model_sky_sampling'
        ):
        self.data = training_data
        
        self.device = device

        self.model = self.create_model(device)
        self.model.load_state_dict(torch.load(save_location))

        
    def predict_spectral_radiance(self, img_tensor):
        self.model.eval()
        tensor = img_tensor.unsqueeze(0).to(self.device)
        pred = self.model(tensor).detach().cpu().reshape(-1)

        return self.data.denormalize(pred)