import os
import torch.nn as nn
from torchvision import transforms
import pickle
import sys
sys.path.append('./')
from models.spade import Generator, Discriminator
from datasets.rwth import load_images_and_poses
from train_loop import train
from misc import setup_fid

if __name__ == "__main__":
    device = 'cuda'

    model_dir = './results/gan/generators_weights/'
    img_dir = './results/gan/generated_images/'
    var_dir = './results/gan/saved_variables/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    if not os.path.exists(var_dir):
        os.makedirs(var_dir)

    print("Creating dataset object")
    # load data
    batch_size = 16
    dims = [64, 64]
    keypoint_pad = 2
    transforms_compose = transforms.Compose([
        transforms.Resize(dims)])
    root = './datasets/rwth/'
    image_datasets, dataloaders, n_classes = load_images_and_poses(batch_size, root, dims[0], transforms_compose, keypoint_pad = keypoint_pad)
    dataset = image_datasets['train']
    loader = dataloaders['train']
    n_keypoints = dataset[0][2].shape[0]
    n_classes = 1
    print("Dataset size: {}".format(len(dataset)))

    # Initialize models
    print("Creating models")
    base_channels = 64
    z_dim = 20
    shared_dim = 128
    generator = Generator(base_channels=base_channels, bottom_width=8, z_dim=z_dim, shared_dim=shared_dim, n_classes=n_classes, c_dim=n_keypoints).to(device)
    discriminator = Discriminator(base_channels=base_channels, n_classes=n_classes, in_channels=n_keypoints+3).to(device)

    # Initialize weights orthogonally
    for module in generator.modules():
        if (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.Embedding)):
            nn.init.orthogonal_(module.weight)
    for module in discriminator.modules():
        if (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.Embedding)):
            nn.init.orthogonal_(module.weight)

    # Setup FID
    print("Calculating FID parameters")
    fid_real_m, fid_real_s = setup_fid(image_datasets['val'], var_dir, batch_size, device)
    print("FID parameters calculated!")

    # Train
    n_epochs = 400
    fid_len = len(image_datasets['val'])
    max_p = 1.0
    generator, discriminator, fids = train(generator, discriminator, loader, fid_real_m, fid_real_s, fid_len, n_epochs = n_epochs, max_p = max_p, 
                                           model_dir = model_dir, img_dir = img_dir, device = device)
    with open(var_dir+'fids.pkl', 'wb') as f:
        pickle.dump(fids, f)