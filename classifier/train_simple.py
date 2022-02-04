from torchvision import transforms
import numpy as np
import os
import pickle
import sys
sys.path.append('./')
from datasets.generated import load_data as load_generated_data
from datasets.rwth import load_data
from train_routine import train

if __name__ == "__main__":
    dims = [64, 64]
    batch_size = 256
    dataset_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(dims),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    reduced_train = None
    generated_reduced_train = None
    generated_dir = './datasets/generated_rwth/'

    # Train hyperparameters
    augmentation_transforms = transforms.Compose([
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomAffine(degrees=10, translate=(0.2, 0.2), scale=(0.8, 1.2)),
        # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    ])
    n_epochs = 50
    lr = 1e-4
    train_with_generated=True
    train_results_dir = './results/classifier/results.pkl'
    if not os.path.exists(train_results_dir):
        os.makedirs(train_results_dir)
    # Mixup
    alpha = 1.0
    beta = 1.0

    # MCCV hyperparameters
    steps = 10
    errs = np.zeros(10)
    MCCV_results_dir = './results/MCCV/'
    if not os.path.exists(MCCV_results_dir):
        os.makedirs(MCCV_results_dir)

    for i in range(steps):
        print("Creating dataset object")
        # Load data in a new order
        image_datasets, dataloaders, n_classes = load_data(batch_size, transforms_compose=dataset_transforms, reduced_train=reduced_train, ntest=0.0, nval=0.2)
        _, generated_dataloaders, _ = load_generated_data(batch_size, dir = generated_dir, transforms_compose=dataset_transforms, reduced_train=generated_reduced_train, ntest=0.0, nval=0.0)

        # Train the model
        model, best_acc = train(image_datasets, dataloaders, n_classes, generated_dataloaders, 
                                n_epochs = n_epochs, augmentation_transforms = augmentation_transforms, 
                                lr = lr, alpha = alpha, beta = beta, train_with_generated=train_with_generated,
                                name='MCCV {}'.format(i), device = 'cuda')
        errs[i] = 1-best_acc
    # Calculate and save error std and mean
    results = {'acc_mean': np.mean(1-errs),
              'acc_std': np.std(1-errs),
              'error_mean': np.mean(errs),
              'error_std': np.std(errs),}
    with open(MCCV_results_dir+'results.pkl', 'wb') as f:
        pickle.dump(results, f)