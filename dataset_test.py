"""
Created on Sat Sep 28 19:59:14 2019
@author: LU
"""
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    """ImageDataset function"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.input = []
        self.label = []
        self.filename = []
        self.transform = transform
        self.num_classes = 0
        for i, _dir in enumerate(sorted(self.root_dir.glob('*'))):
            for file in _dir.glob('*'):
                self.input.append(file)
                self.label.append(i)
                self.filename.append(str(file)[21:])

            self.num_classes += 1

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        image = Image.open(self.input[index]).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, self.label[index], self.filename[index]
