import os
import torch
from torchvision import transforms
from torchvision.utils import save_image
import sys
sys.path.append('./')
from models.spade import Generator
from datasets.rwth import load_images_and_poses

if __name__ == "__main__":
    device = 'cuda'
    # load data

    model_dir = './results/gan/generators_weights/'
    img_dir = './results/generated_new_images/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    batch_size = 1
    dims = [64, 64]
    transforms_compose = transforms.Compose([
        # transforms.ToTensor(),
        transforms.Resize(dims)])
    print("Creating dataset object")
    root = './datasets/rwth/'
    image_datasets, dataloaders, n_classes = load_images_and_poses(batch_size, root, dims[0], transforms_compose)
    dataset = image_datasets['train']
    loader = dataloaders['train']
    n_keypoints = dataset[0][2].shape[0]
    print(dataset.dataset.dataset.classes)

    # Initialize models
    print("Creating models")
    base_channels = 64
    z_dim = 20
    n_classes = 1
    shared_dim = 128
    generator = Generator(base_channels=base_channels, bottom_width=8, z_dim=z_dim, shared_dim=shared_dim, n_classes=n_classes, c_dim=n_keypoints).to(device)
    generator.load_state_dict(torch.load(model_dir+'gen.state_dict'))
    generator.eval()

    n_gen_images = 10

    for batch_ndx, sample in enumerate(loader):
        real, label, pose = sample[0], sample[1], sample[2]
        if not os.path.exists(img_dir+str(label[0].numpy())+"/"):
            os.makedirs(img_dir+str(label[0].numpy())+"/")
        print('class: {}'.format(str(label[0].numpy())))
        y = pose.to(device)
        z = torch.clamp(torch.randn(batch_size, z_dim, device=device), min=-0.4, max=0.4)      # Generate random noise (z)
        fake = generator(z, y)
        save_image(fake, img_dir+str(label[0].numpy())+"/{}{}.png".format(str(label[0].numpy()), batch_ndx))