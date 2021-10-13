from torch.utils import data
import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import random_split, ConcatDataset
import torchvision.models as models
import copy
import random

import sys
sys.path.append('./')
from datasets.autsl import load_data

if __name__ == "__main__":
    device = 'cuda'

    batch_size = 164
    dims = [64, 64]

    print("Creating dataset object")
    transforms_compose = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(dims),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    augmentation_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=10, translate=(0.2, 0.2), scale=(0.8, 1.2)),
        # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    ])
    # load data
    dir = './datasets/autsl/poses/'
    image_datasets, dataloaders = load_data(batch_size, dir=dir, transforms_compose=transforms_compose)
    # n_classes = len(image_datasets['train'].classes)
    n_classes = 226
    dataset_sizes = {'train': len(image_datasets['train']),
                     'val': len(image_datasets['val'])}
    """
    dir = './datasets/autsl/poses/'
    small_dataset = False
    with_generated = False
    datasets_list = []
    for signer in os.listdir(dir):
        dataset = datasets.ImageFolder(root=dir+signer+'/', transform=transforms_compose)
        if small_dataset:
            new_dataset_size = n_classes*10
            discarded_size = len(dataset)-new_dataset_size
            dataset, _ = random_split(dataset, [new_dataset_size, discarded_size], generator=torch.Generator().manual_seed(42))
        datasets_list.append(dataset)
    random.Random(42).shuffle(datasets_list)
    n_classes = len(datasets_list[0].classes)

    test_size = int(0.25 * len(datasets_list))
    val_size = int(0.2 * (len(datasets_list) - test_size))
    train_size = len(datasets_list) - test_size - val_size

    test_dataset = ConcatDataset(datasets_list[:test_size])
    val_dataset = ConcatDataset(datasets_list[test_size:test_size+val_size])
    train_dataset = ConcatDataset(datasets_list[test_size+val_size:])

    if with_generated:
        generated_dataset = datasets.ImageFolder(root='./results/generated_new_images/', transform=transforms_compose)
        generated_train_size = int(0.2*len(train_dataset))
        generated_discarded_size = len(generated_dataset)-generated_train_size

        generated_train_dataset, _ = random_split(generated_dataset, [generated_train_size, generated_discarded_size], generator=torch.Generator().manual_seed(42))
        train_dataset = ConcatDataset([train_dataset, generated_train_dataset]) 

    dataloaders = {'train': data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=4),
                   'val': data.DataLoader(val_dataset, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=4)}
    dataset_sizes = {'train': len(train_dataset),
                   'val': len(val_dataset)}
    """

    # Initialize models
    print("Creating models")
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, n_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Initialize optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.0, 0.999), eps=1e-6)

    ###########
    #  TRAIN  #
    ###########
    print("Training started")
    n_epochs = 30
    cur_step = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(n_epochs):
        print('##############################')
        print('#epoch: {}'.format(epoch))

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                if phase == 'train':
                    inputs = augmentation_transforms(inputs).to(device)
                else:
                    inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)