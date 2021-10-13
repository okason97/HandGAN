import os
import torch
from torchvision import transforms
from torchvision.utils import save_image
import math
import sys
import numpy as np
sys.path.append('./')
from models.spade import Generator
from datasets.autsl import load_images_and_poses

class GeneratorSampler():
    def __init__(self, generator, batch_size, z_dim, device):
        self.z_dim = z_dim
        self.generator = generator
        self.batch_size = batch_size
        self.device = device

    def get_samples(self, y, sample_size):
        fakes = torch.tensor([], device='cpu')
        y_emb = self.generator.shared_emb(torch.tensor([y]*batch_size, dtype=torch.long, device=self.device))
        for _ in range(sample_size):
            # Generate random noise (z)
            z = torch.clamp(torch.randn(self.batch_size, self.z_dim, device=self.device), min=-0.4, max=0.4)      
            fake = self.generator(z, y_emb)
            fakes = torch.cat((fakes, fake.to('cpu')))
        return fakes

    def get_samples_spade(self, poses_loader, sample_size):
        fakes = torch.tensor([], device='cpu')
        for _ in range(sample_size):
            # Generate random noise (z)
            z = torch.clamp(torch.randn(self.batch_size, self.z_dim, device=self.device), min=-0.4, max=0.4)      
            _, pose = next(iter(poses_loader))
            fake = self.generator(z, pose.to(self.device))
            fakes = torch.cat((fakes, fake.to('cpu')))
        return fakes

if __name__ == "__main__":
    device = 'cuda'
    # load data

    model_dir = './results/generators_weights/'
    img_dir = './results/birthday_images/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    
    # load data
    print("Creating dataset object")
    batch_size = 1
    dims = [64, 64]
    transforms_compose = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(dims)])
    dir_images = './datasets/autsl/cropped_images/'
    dir_poses = './datasets/autsl/poses/'
    image_datasets, dataloaders = load_images_and_poses(batch_size, transforms_compose, dir_images, dir_poses)
    poses_loader = dataloaders['test']
    n_classes = 226

    # Initialize models
    print("Creating models")
    base_channels = 64
    z_dim = 20
    shared_dim = 128
    generator = Generator(base_channels=base_channels, bottom_width=8, z_dim=z_dim, shared_dim=shared_dim, n_classes=n_classes).to(device)
    generator.load_state_dict(torch.load(model_dir+'gen.state_dict'))
    generator.eval()

    # Birthday paradox
    sample_size = 20
    n_pairs = 5
    sampler = GeneratorSampler(generator, batch_size, z_dim, device)

    mode = 'pose_conditioned'
    if mode == 'class_conditioned':
        for y in range(n_classes):
            print('class: {}'.format(y))
            samples = sampler.get_samples(y, sample_size)

            # Calculate closest pairs
            min_dist = [math.inf for _ in range(n_pairs)]
            min_pair = [None for _ in range(n_pairs)]
            for i in range(len(samples)-1):
                for j in range(len(samples)-1-i):
                    max_idx = np.argmax(min_dist)
                    dist = torch.dist(samples[i], samples[i+j+1])
                    if torch.dist(samples[i], samples[i+j+1]) < min_dist[max_idx]:
                        min_dist[max_idx] = dist
                        min_pair[max_idx] = [samples[i], samples[i+j+1]]

            # Save closest pairs
            if not os.path.exists(img_dir+y):
                os.makedirs(img_dir+y)
            for i in range(len(min_pair)):
                save_image(min_pair[i], img_dir+y+"/pair{}.png".format(i))
    elif mode == 'pose_conditioned':
        for n in range(10):
            print('iteration: {}'.format(n))

            samples = sampler.get_samples_spade(poses_loader, sample_size)

            # Calculate closest pairs
            min_dist = [math.inf for _ in range(n_pairs)]
            min_pair = [None for _ in range(n_pairs)]
            for i in range(len(samples)-1):
                for j in range(len(samples)-1-i):
                    max_idx = np.argmax(min_dist)
                    dist = torch.dist(samples[i], samples[i+j+1])
                    if torch.dist(samples[i], samples[i+j+1]) < min_dist[max_idx]:
                        min_dist[max_idx] = dist
                        min_pair[max_idx] = [samples[i], samples[i+j+1]]

            # Save closest pairs
            if not os.path.exists(img_dir+str(n)):
                os.makedirs(img_dir+str(n))
            for i in range(len(min_pair)):
                save_image(min_pair[i], img_dir+str(n)+"/pair{}.png".format(i))