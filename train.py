"""
@author: LU
"""
import copy
from pathlib import Path
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms, models
import torch
from dataset import ImageDataset

##REPRODUCIBILITY
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DATASET_ROOT = '../dataset/train'

def train():
    """train function"""
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    train_set = ImageDataset(Path(DATASET_ROOT), data_transform)
    data_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True, num_workers=0)
    model = models.resnet152(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 13)
    model = model.cuda()
    model.train()
    best_model_params = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    num_epochs = 100
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.0005, momentum=0.9)

    for epoch in range(num_epochs):
        print(f'Epoch: {epoch + 1}/{num_epochs}')
        print('-' * len(f'Epoch: {epoch + 1}/{num_epochs}'))
        training_loss = 0.0
        training_corrects = 0

        for inputs, labels in data_loader:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
            optimizer.zero_grad()
            outputs = model(inputs)
            preds = torch.max(outputs.data, 1)[1]
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            training_loss += loss.item() * inputs.size(0)
            training_corrects += torch.sum(preds == labels.data)

        training_loss = training_loss / len(train_set)
        training_acc = training_corrects.double() / len(train_set)
        print(f'Training loss: {training_loss:.4f}\taccuracy: {training_acc:.4f}\n')

        if training_acc > best_acc:
            best_acc = training_acc
            best_model_params = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_model_params)
    torch.save(model, f'model-{best_acc:.2f}-best_train_acc.pth')


if __name__ == '__main__':
    train()
