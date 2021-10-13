import handshape_datasets as hd
from torchvision import transforms
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class HandshapeDataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images = Tensor(images)
        self.labels = Tensor(labels).long()
        self.classes = self.labels.unique()
        if transform is None:
            transform = transforms.Compose([])
        self.transform = transform

    def __getitem__(self, index):
        return self.transform(self.images[index]), self.labels[index]

    def __len__(self):
        return self.images.size(0)

def load_data(batch_size, transforms_compose=None, seed=42, ntest=0.2, nval=0.25):
    
    images, metadata = hd.load("rwth")
    x_ = images
    y_ = metadata['y']
    unique_y = np.unique(y_, return_counts=True)
    min_images = 10
    arr_x = []
    arr_y = []
    for i in range(len(images)):
        if unique_y[1][y_[i]]>min_images:
            arr_x.append(x_[i])
            arr_y.append(y_[i])
    x_ = np.array(arr_x)
    y_ = np.array(arr_y)
    unique_y = np.unique(y_)
    for i in range(len(y_)):
        y_[i] = np.where(unique_y == y_[i])[0][0]
    
    x_, X_test, y_, y_test = train_test_split(x_, y_, test_size=ntest, random_state=seed, stratify=y_)
    X_train, X_val, y_train, y_val = train_test_split(x_, y_, test_size=nval, random_state=seed, stratify=y_)

    test_dataset = HandshapeDataset(X_test.transpose(0,3,1,2), y_test, transform=transforms_compose)
    val_dataset = HandshapeDataset(X_val.transpose(0,3,1,2), y_val, transform=transforms_compose)
    train_dataset = HandshapeDataset(X_train.transpose(0,3,1,2), y_train, transform=transforms_compose)

    dataloaders = {'train': DataLoader(train_dataset, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=4),
                   'val': DataLoader(val_dataset, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=4),
                   'test': DataLoader(test_dataset, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=4)}
    image_datasets = {'train': train_dataset,
                     'val': val_dataset,
                     'test': test_dataset}
    return image_datasets, dataloaders