import  pytorch_fid_wrapper as pfw
import torch
from torchvision import transforms
from torchvision.utils import save_image
from random import randrange
from math import inf
import sys
sys.path.append('./')
from misc import ortho, RandomApplyEach

def train(generator, discriminator, loader, fid_real_m, fid_real_s, fid_len, augmentation_transforms = None, n_epochs = 100, D_steps = 2, ada_target = 0.6, adjustment_size = 500000, max_p = 0.0, 
          ortho_reg = False, ortho_strength = 1e-4, model_dir = './results/generators_weights/', img_dir = None, device = 'cuda'):
    print("Training started")
    # ADA
    # set max_p to 0 to disable
    if augmentation_transforms is None:
        augmentation_transforms = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.8, 1.2)),
            # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        ]
    p = torch.tensor(0.0, device=device)
    update_iteration = 8
    augmentation = RandomApplyEach(augmentation_transforms, p).to(device)
    ada_buf = torch.tensor([0.0, 0.0], device=device)    

    # Initialize optimizers
    gen_opt = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.0, 0.999), eps=1e-6)
    disc_opt = torch.optim.Adam(discriminator.parameters(), lr=4e-4, betas=(0.0, 0.999), eps=1e-6)

    cur_step = 0
    min_fid = inf

    # Generate random noise (z)
    fixed_image, _, fixed_pose = next(iter(loader))
    fixed_z = torch.randn(len(fixed_image), generator.z_dim, device=device)
    fixed_y = fixed_pose
    if not img_dir is None:
        save_image(fixed_image, img_dir+"real.png")
    # Generate a batch of poses (y)
    fixed_y = fixed_y.to(device)

    fakes = torch.tensor([], device='cpu')
    fids = torch.tensor([], device='cpu')

    fid_step = 0

    for epoch in range(n_epochs):
        print('##############################')
        print('#epoch: {}'.format(epoch))

        for _, sample in enumerate(loader):
            real, _, pose = sample[0], sample[1], sample[2]
            _y = pose

            batch_size = len(real)
            real = real.to(device)

            disc_opt.zero_grad()
            gen_opt.zero_grad()

            for _ in range(D_steps):
                # Zero out the discriminator gradients
                disc_opt.zero_grad()
                ### Update discriminator ###
                # Get noise corresponding to the current batch_size 
                z = torch.randn(batch_size, generator.z_dim, device=device)       # Generate random noise (z)
                y = _y.to(device)                            # Generate a batch of labels (y), one for each class
                fake = generator(z, y)
                """
                fake = augmentation(fake.detach())
                real_augmented = augmentation(real)
                """
                # fake_augmented, real_augmented = torch.split(augmentation(torch.cat([fake, real], dim=0)), fake.shape[0])
                seed = randrange(2**64)
                torch.manual_seed(seed)
                fake_augmented = augmentation(fake)
                torch.manual_seed(seed)
                real_augmented = augmentation(real)
                torch.manual_seed(seed)
                y_augmented = augmentation(y)

                disc_fake_pred = discriminator(torch.cat([fake_augmented, y_augmented], dim=1))
                disc_real_pred = discriminator(torch.cat([real_augmented, y_augmented], dim=1))

                ada_buf += torch.tensor(
                    (torch.clamp(torch.sign(disc_real_pred), min=0, max=1).sum().item(), disc_real_pred.shape[0]),
                    device=device
                )

                #loss
                disc_loss = discriminator.loss(disc_fake_pred, disc_real_pred)
                # Update gradients
                disc_loss.backward()

                if ortho_reg:
                    ortho(discriminator, ortho_strength)

                # Update optimizer
                disc_opt.step()

            ### Update generator ###
            # Zero out the generator gradients
            gen_opt.zero_grad()

            fake = generator(z, y)
            seed = randrange(2**64)
            torch.manual_seed(seed)
            fake_augmented = augmentation(fake)
            torch.manual_seed(seed)
            y_augmented = augmentation(y)
            disc_fake_pred = discriminator(torch.cat([fake_augmented, y_augmented], dim=1))
            #loss
            gen_loss =  generator.loss(disc_fake_pred)
            # Update gradients
            gen_loss.backward()

            if ortho_reg:
                ortho(generator, ortho_strength)

            # Update optimizer
            gen_opt.step()

            fakes = torch.cat((fakes, fake.to('cpu')))

            if cur_step % update_iteration == 0:
                # Adaptive Data Augmentation
                pred_signs, n_pred = ada_buf
                r_t = pred_signs / n_pred

                sign = r_t - ada_target

                augmentation.p = torch.clamp(augmentation.p + (sign * n_pred / adjustment_size), min=0, max=max_p)

                ada_buf = ada_buf * 0

            cur_step +=1

            if cur_step*batch_size/(fid_step+1) > fid_len:
                fid_step += 1
                print('===========================================================================')
                val_fid = pfw.fid(fakes, real_m=fid_real_m, real_s=fid_real_s)
                fids = torch.cat((fids, torch.tensor([val_fid])))
                fakes = torch.tensor([], device='cpu')
                print('FID: {}'.format(val_fid))
                print('Augmentation p: {}'.format(augmentation.p))
                if (val_fid < min_fid):
                    min_fid = val_fid
                    torch.save(generator.state_dict(), (model_dir+'gen.state_dict'))
                    torch.save(discriminator.state_dict(), (model_dir+'disc.state_dict'))
                print('===========================================================================')

        print('saved images')
        if not img_dir is None:
            fake = generator(fixed_z, fixed_y)
            save_image(fake, img_dir+"generated{}.png".format(epoch))

    generator.load_state_dict(torch.load(model_dir+'gen.state_dict'))
    discriminator.load_state_dict(torch.load(model_dir+'disc.state_dict'))

    return generator, discriminator, fids