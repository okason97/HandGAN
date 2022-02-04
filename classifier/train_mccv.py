from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
import numpy as np
import os
import pickle
import sys
sys.path.append('./')
from datasets.generated import load_data as load_generated_data
from datasets.rwth import load_data
from train_loop import train
from test_loop import test

if __name__ == "__main__":
    device = 'cuda'
    dims = [64, 64]
    batch_size = 256
    dataset_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(dims),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    reduced_train = None
    generated_reduced_train = None
    generated_dir = './datasets/generated_rwth_spade_ada/'

    # Train hyperparameters
    augmentation_transforms = transforms.Compose([
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomAffine(degrees=10, translate=(0.2, 0.2), scale=(0.8, 1.2)),
        # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    ])
    n_epochs = 50
    lr = 1e-4
    train_with_generated=True
    train_results_dir = './results/classifier/'
    if not os.path.exists(train_results_dir):
        os.makedirs(train_results_dir)
    # Mixup
    alpha = 1.0
    beta = 1.0
    g, generated_dataloaders, _ = load_generated_data(batch_size, dir = generated_dir, transforms_compose=dataset_transforms, reduced_train=generated_reduced_train, ntest=0.0, nval=0.0)

    # MCCV hyperparameters
    steps = 1
    errs = np.zeros(steps)
    MCCV_results_dir = './results/MCCV/'
    if not os.path.exists(MCCV_results_dir):
        os.makedirs(MCCV_results_dir)

    for i in range(steps):
        print("MCCV step {}".format(i))
        print("Creating dataset object")
        # Load data in a new order
        image_datasets, dataloaders, n_classes = load_data(batch_size, transforms_compose=dataset_transforms,
                                                           reduced_train=reduced_train, ntest=0.2, nval=0.2, seed=42)

        # Initialize models
        print("Creating models")
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, n_classes)
        model = model.to(device)

        # Train the model
        model, train_results = train(model, image_datasets, dataloaders, generated_dataloaders, 
                                n_epochs = n_epochs, augmentation_transforms = augmentation_transforms, 
                                lr = lr, alpha = alpha, beta = beta, train_with_generated=train_with_generated,
                                name='MCCV {}'.format(i), device = device)

        print('Best val Acc: {:4f}'.format(np.max(train_results['val']['epoch_acc'])))
        if not train_results_dir is None:
            with open(train_results_dir+'results{}.pkl'.format(i), 'wb') as f:
                pickle.dump(train_results, f)

        test_results = test(image_datasets, dataloaders, model, device = device)

        errs[i] = 1-test_results['test']['acc']

    # Calculate and save error std and mean
    results = {'acc_mean': np.mean(1-errs),
              'acc_std': np.std(1-errs),
              'err_mean': np.mean(errs),
              'err_std': np.std(errs),}
    with open(MCCV_results_dir+'results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print('Mean Acc: {:4f}, Std Acc: {:4f}'.format(results['acc_mean'], results['acc_std']))
    print('Mean Err: {:4f}, Std Err: {:4f}'.format(results['err_mean'], results['err_std']))