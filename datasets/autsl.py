from torchvision import datasets, transforms
from torch import zeros
from torch.utils.data import Dataset, TensorDataset, DataLoader, ConcatDataset
import os
import json 

class ImagePoseDataset(Dataset):
    def __init__(self, images_dataset, poses_dataset, transforms_compose):
        self.images_dataset = images_dataset
        self.poses_dataset = poses_dataset
        self.transforms_compose = transforms_compose

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        return self.transforms_compose(self.images_dataset[idx][0]), self.images_dataset[idx][1], self.transforms_compose(self.poses_dataset[idx][0])

    def __len__(self):
        return len(self.images_dataset)

def load_data(batch_size, transforms_compose=None, dir='./autsl/cropped_images/', seed=42, ntest=0.25, nval=0.2):
    if transforms_compose is None:
        transforms_compose = transforms.Compose([
            transforms.ToTensor(),])
    datasets_list = []
    for signer in os.listdir(dir):
        dataset = datasets.ImageFolder(root=dir+signer+'/', transform=transforms_compose)
        datasets_list.append(dataset)
    # random.Random(seed).shuffle(datasets_list)

    test_size = int(ntest * len(datasets_list))
    val_size = int(nval * (len(datasets_list) - test_size))

    test_dataset = ConcatDataset(datasets_list[:test_size])
    val_dataset = ConcatDataset(datasets_list[test_size:test_size+val_size])
    train_dataset = ConcatDataset(datasets_list[test_size+val_size:])

    dataloaders = {'train': DataLoader(train_dataset, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=4),
                   'val': DataLoader(val_dataset, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=4),
                   'test': DataLoader(test_dataset, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=4)}
    image_datasets = {'train': train_dataset,
                     'val': val_dataset,
                     'test': test_dataset}
    return image_datasets, dataloaders

def load_images_and_poses(batch_size, transforms_compose=None, dir_images='./autsl/cropped_images/', dir_poses='./autsl/poses/', ntest=0.25, nval=0.2):
    if transforms_compose is None:
        transforms_compose = transforms.Compose([])
    
    datasets_list = []
    list_dir_images = os.listdir(dir_images)
    list_dir_poses = os.listdir(dir_poses)
    min_confidence = 0.4
    pad = 5
    for i in range(len(list_dir_images)):
        images_dataset = datasets.ImageFolder(root=dir_images+list_dir_images[i]+'/', transform=transforms.ToTensor())

        poses = []
        for pose_file_dir in os.listdir(dir_poses+list_dir_poses[i]+'/'):
            f = open(dir_poses+list_dir_poses[i]+'/'+pose_file_dir)
            pose_data = json.load(f)['people'][0]
            n_pose_keypoints = int(len(pose_data['pose_keypoints_2d'])/3)
            n_hand_left_keypoints = int(len(pose_data['hand_left_keypoints_2d'])/3)
            n_hand_right_keypoints = int(len(pose_data['hand_right_keypoints_2d'])/3)
            pose_map = zeros(n_pose_keypoints+n_hand_left_keypoints+n_hand_right_keypoints, images_dataset[i][0].shape[1], images_dataset[i][0].shape[2])
            keypoints = pose_data['pose_keypoints_2d'] + pose_data['hand_left_keypoints_2d'] + pose_data['hand_right_keypoints_2d']
            for j in range(n_pose_keypoints+n_hand_left_keypoints+n_hand_right_keypoints):
                confidence = keypoints[j*3+2]
                if confidence>min_confidence:
                    x = round(keypoints[j*3])
                    y = round(keypoints[j*3+1])
                    pose_map[j, max(0,x-pad):min(256,x+pad), max(0,y-pad):min(256,y+pad)] = 256*min(confidence, 1)
            f.close()
            poses.append(pose_map)
        poses_dataset = TensorDataset(poses)
        dataset = ImagePoseDataset(images_dataset, poses_dataset, transform=transforms_compose)
        datasets_list.append(dataset)

    test_size = int(ntest * len(datasets_list))
    val_size = int(nval * (len(datasets_list) - test_size))

    test_dataset = ConcatDataset(datasets_list[:test_size])
    val_dataset = ConcatDataset(datasets_list[test_size:test_size+val_size])
    train_dataset = ConcatDataset(datasets_list[test_size+val_size:])

    dataloaders = {'train': DataLoader(train_dataset, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=4),
                   'val': DataLoader(val_dataset, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=4),
                   'test': DataLoader(test_dataset, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=4)}
    image_datasets = {'train': train_dataset,
                     'val': val_dataset,
                     'test': test_dataset}
    return image_datasets, dataloaders