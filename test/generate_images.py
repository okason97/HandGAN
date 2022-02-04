import os
import torch
from torchvision import transforms
from torchvision.utils import save_image
import sys
sys.path.append('./')
from models.ada_gan import Generator
from datasets.rwth import load_data

if __name__ == "__main__":
    device = 'cuda'
    # load data

    model_dir = './results/generators_weights/'
    img_dir = './results/generated_new_images/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    batch_size = 1
    dims = [64, 64]
    transforms_compose = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(dims)])
    print("Creating dataset object")
    image_datasets, dataloaders, n_classes = load_data(batch_size, transforms_compose)
    dataset = image_datasets['train']
    loader = dataloaders['train']

    # Initialize models
    print("Creating models")
    base_channels = 64
    z_dim = 120
    shared_dim = 128
    generator = Generator(base_channels=base_channels, bottom_width=8, z_dim=z_dim, shared_dim=shared_dim, n_classes=n_classes).to(device)
    generator.load_state_dict(torch.load(model_dir+'gen.state_dict'))
    generator.eval()

    n_gen_images = 100

    for y in range(n_classes):
        if not os.path.exists(img_dir+str(dataset.classes.numpy()[y])+"/"):
            os.makedirs(img_dir+str(dataset.classes.numpy()[y])+"/")
        print('class: {}'.format(str(dataset.classes.numpy()[y])))
        y_emb = generator.shared_emb(torch.tensor([y]*batch_size, dtype=torch.long, device=device))
        for n_gen in range(n_gen_images):

            z = torch.clamp(torch.randn(batch_size, z_dim, device=device), min=-0.4, max=0.4)      # Generate random noise (z)
            fake = generator(z, y_emb)
            save_image(fake, img_dir+str(dataset.classes.numpy()[y])+"/{}{}.png".format(str(dataset.classes.numpy()[y]),n_gen))