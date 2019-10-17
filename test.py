"""@author: LU"""
import csv
from pathlib import Path
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset_test import ImageDataset
CUDA_DEVICES = 0
DATASET_ROOT = '../dataset/test'
CLASS_ = '../dataset/train'
PATH_TO_WEIGHTS = './model-1.00-best_train_acc.pth'

def test():
    """test function"""
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_set = ImageDataset(Path(DATASET_ROOT), data_transform)
    data_loader = DataLoader(
        dataset=test_set, batch_size=1, shuffle=False, num_workers=1)
    classes = [_dir.name for _dir in Path(CLASS_).glob('*')]
    model = torch.load(PATH_TO_WEIGHTS)
    model = model.cuda(CUDA_DEVICES)
    model.eval()
    table = [['id', 'label']]
    i = 0
    with torch.no_grad():
        for inputs, labels, filename in data_loader:
            i += 1
            print('i:', i)
            inputs = Variable(inputs.cuda(CUDA_DEVICES))
            labels = Variable(labels.cuda(CUDA_DEVICES))
            outputs = model(inputs)
            predicted = torch.max(outputs.data, 1)[1]
            pre = predicted.item()
            table.append([str(filename)[2:12], classes[pre]])

    with open('test.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        table.sort()
        writer.writerows(table)


if __name__ == '__main__':
    test()
    