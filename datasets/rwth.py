import handshape_datasets as hd
from torchvision import transforms
import numpy as np
from torch import zeros, tensor, manual_seed
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
import skimage.io as io
import json
import os
from skimage.draw import disk

class HandshapeDataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = tensor(labels).long()
        self.classes = self.labels.unique()
        if transform is None:
            transform = transforms.Compose([])
        self.transform = transform

    def __getitem__(self, index):
        return self.transform(self.images[index]), self.labels[index]

    def __len__(self):
        return self.images.shape[0]

class RWTHPoseDataset(Dataset):
    def __init__(self, root = './', shape = 64, transform = None, keypoint_pad = 2):
        self.root = root
        self.shape = shape
        if transform is None:
            transform = transforms.Compose([])
        self.transform = transform
        self.keypoint_pad = keypoint_pad
        self.le = None
        
        anns = pd.read_csv(self.root + 'annotations.csv', sep=' ', header=None)

        self.images, self.labels, self.hand_keypoints = self.load(anns)
        self.classes = np.unique(self.labels)

    def load(self, anns):
        paths, full_labels = anns[0], anns[1]
        
        self.le = preprocessing.LabelEncoder()
        self.le.fit(full_labels)
        full_labels = self.le.transform(full_labels)

        images = []
        hand_keypoints = []
        labels = []
        for path, label in zip(paths, full_labels):
            base_path = '/'.join(os.path.normpath(path).split(os.sep)[-3:]).replace('*','_')
            img_path = self.root + 'images/' + base_path
            pose_path = self.root + 'poses/' + base_path[:-4]+'_keypoints.json'
            if os.path.exists(img_path):
                # Load pose
                f = open(pose_path)
                pose = json.load(f)
                f.close()

                if len(pose['people']) > 0:
                    if sum(pose['people'][0]['hand_right_keypoints_2d']) > 0:
                        # Load image
                        image = io.imread(img_path)
                        images.append(image)
                        labels.append(label)
                        hand_pose = pose['people'][0]['hand_right_keypoints_2d']
                        hand_keypoints.append(self.load_keypoints(hand_pose))
                    elif sum(pose['people'][0]['hand_left_keypoints_2d']) > 0:
                        # Load image
                        image = io.imread(img_path)
                        images.append(image)
                        labels.append(label)
                        hand_pose = pose['people'][0]['hand_left_keypoints_2d']
                        hand_keypoints.append(self.load_keypoints(hand_pose))
        return np.array(images), np.array(labels), hand_keypoints

    def load_keypoints(self, hand_pose):
        n_pose_keypoints = int(len(hand_pose)/3)
        pose_map = zeros((n_pose_keypoints, self.shape, self.shape))

        for j in range(n_pose_keypoints):
            rr, cc = disk((hand_pose[j*3+1], hand_pose[j*3]), self.keypoint_pad, shape=(self.shape, self.shape))
            pose_map[j, rr, cc] = 1.0
        return pose_map

    def __getitem__(self, index):
        image = tensor(np.moveaxis(self.images[index], -1, 0))/255
        return self.transform(image), self.labels[index], self.hand_keypoints[index]

    def __len__(self):
        return self.images.shape[0]

def load_data(batch_size, transforms_compose=None, seed=42, ntest=0.2, nval=0.2, reduced_train=None):

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

    if not (reduced_train is None):
        X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=1-reduced_train, random_state=seed, stratify=y_train)

    test_dataset = HandshapeDataset(X_test, y_test, transform=transforms_compose)
    val_dataset = HandshapeDataset(X_val, y_val, transform=transforms_compose)
    train_dataset = HandshapeDataset(X_train, y_train, transform=transforms_compose)

    dataloaders = {'train': DataLoader(train_dataset, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=4),
                   'val': DataLoader(val_dataset, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=4),
                   'test': DataLoader(test_dataset, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=4)}
    image_datasets = {'train': train_dataset,
                     'val': val_dataset,
                     'test': test_dataset}
    return image_datasets, dataloaders, len(unique_y)

def load_images_and_poses(batch_size, root='./', shape = 64, transforms_compose=None, seed=42, nval=0.2, ntest=0.2, keypoint_pad = 2):
    manual_seed(seed)
    train_dataset = RWTHPoseDataset(root, shape, transform=transforms_compose, keypoint_pad = keypoint_pad)
    n_classes = len(train_dataset.classes)

    if ntest>0:
        """
        train_idx, test_idx, targets, _= train_test_split(
            np.arange(len(targets)), targets, test_size=ntest, random_state=seed, shuffle=True)
        """
        train_size = int((1-ntest) * len(train_dataset))
        test_size = len(train_dataset) - train_size
        train_dataset, test_dataset = random_split(train_dataset, [train_size, test_size])
        """
        train_dataset = Subset(train_dataset, train_idx)
        test_dataset = Subset(train_dataset, test_idx)
        """
    if nval>0:
        """
        train_idx, valid_idx= train_test_split(
            np.arange(len(targets)), test_size=ntest, random_state=seed, shuffle=True)
        """
        train_size = int((1-nval) * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        """
        train_dataset = Subset(train_dataset, train_idx)
        val_dataset = Subset(train_dataset, valid_idx)
        """

    dataloaders = {'train': DataLoader(train_dataset, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=4),
                   'val': DataLoader(val_dataset, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=4)
                          if nval>0 else None,
                   'test': DataLoader(test_dataset, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=4)
                           if ntest>0 else None}
    image_datasets = {'train': train_dataset,
                     'val': val_dataset if nval>0 else None,
                     'test': test_dataset if ntest>0 else None}
    return image_datasets, dataloaders, n_classes