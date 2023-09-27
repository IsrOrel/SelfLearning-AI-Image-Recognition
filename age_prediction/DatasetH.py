import torch
import pandas as pd
import os
from PIL import Image
import skimage as io
from torch.utils.data import Dataset
data = pd.read_csv("C:/Bachelor Of Science/ImageClassification/Age detection intenet model/train.csv")
data['Class'] = data['Class'].map({'YOUNG': 0, 'MIDDLE': 1, 'OLD': 2})
print(data)

class AgeDetection(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.__annotations__ = data
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.__annotations__)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.__annotations__.iloc[index, 0])
        image = Image.open(img_path)
        y_label = torch.tensor(int(self.__annotations__.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)
        return (image, y_label)

    def resize_image(self, image):
        # Resize the image to a consistent size (e.g., 128x128)
        image = image.resize((128, 128))
        return image
