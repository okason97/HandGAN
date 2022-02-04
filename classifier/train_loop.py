import torch
import torch.nn as nn
import copy
import numpy as np
from math import inf
from torchvision import transforms
import sys
sys.path.append('./')
from misc import mixup_data, mixup_criterion, sigmoid

def train(model, image_datasets, dataloaders, generated_dataloaders, n_epochs = 50,
          augmentation_transforms = None, lr=1e-4, alpha = 1.0, beta = 1.0, train_with_generated=True, 
          name='', device = 'cuda'):
    print("Training started")
    if augmentation_transforms == None:
        augmentation_transforms = transforms.Compose([
            # transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomAffine(degrees=10, translate=(0.2, 0.2), scale=(0.8, 1.2)),
            # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        ])

    dataset_sizes = {'train': len(image_datasets['train']),
                     'val': len(image_datasets['val'])}

    criterion = nn.CrossEntropyLoss()

    # Initialize optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.0, 0.999), eps=1e-6)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = inf

    results = {
        'train': {
            'epoch_loss': np.zeros(n_epochs),
            'epoch_acc': np.zeros(n_epochs),
        },
        'val': {
            'epoch_loss': np.zeros(n_epochs),
            'epoch_acc': np.zeros(n_epochs),
        },
    }

    confidence_alpha = -3.
    confidence_beta = 0.5
    confidence_max = 1.

    for epoch in range(n_epochs):
        print('##############################')
        print('#{} epoch: {}'.format(name, epoch))

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
                    if train_with_generated:
                        inputs_g, labels_g = next(iter(generated_dataloaders[phase]))
                        inputs_g, lam_g = mixup_data(inputs_g[:inputs.shape[0], :], inputs, alpha, beta)
                        inputs_g = augmentation_transforms(inputs).to(device)
                        labels_ga = labels_g[:labels.shape[0]].to(device)
                        labels_gb = labels.to(device)
                    idx = torch.randperm(inputs.shape[0])
                    inputs, lam = mixup_data(inputs[idx], inputs, alpha, beta)
                    labels_a = labels[idx].to(device)
                    labels_b = labels.to(device)
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
                    if train_with_generated:
                        outputs_g = model(inputs_g)
                    _, preds = torch.max(outputs, 1)
                    if phase == 'train':
                        loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                        if train_with_generated:
                            loss_g = mixup_criterion(criterion, outputs_g, labels_ga, labels_gb, lam_g)
                            loss = loss + sigmoid(confidence_alpha+epoch*confidence_beta)*confidence_max*loss_g
                    else:
                        loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                if phase == 'train':
                    running_corrects += torch.sum(torch.logical_or(preds == labels_a.data,preds == labels_b.data))
                else:
                    running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            results[phase]['epoch_loss'][epoch] = epoch_loss
            results[phase]['epoch_acc'][epoch] = epoch_acc

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, results[phase]['epoch_loss'][epoch], results[phase]['epoch_acc'][epoch]))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, results