import numpy as np
import torch
import yaml
from mnist import Mnist
from torch.utils.data import DataLoader
import logging
from vanilla_vae import VanillaVAE
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    #print(' config : {0}'.format(config))
        
    # Load data
    mnist_data = Mnist(data_dir=config['dataset_dir'], normalize=True, mode='Train')
    batch_size = config['batch_size']
    lr = config['lr']
    device = config['device']
    mnist_data_loader = DataLoader(mnist_data, batch_size=batch_size)

    # Load vae model
    vvae = VanillaVAE(in_channels=1, width=28, height=28, dim_latent=72, hidden_dims=[32,64,128]).to(device=device)
    #criterian and optimizer
    criterion = vvae.loss_func
    optimizer = torch.optim.Adam(vvae.parameters(), lr=lr)

    for epoch in range(config['num_epoch']):
        for i, data_item in enumerate(mnist_data_loader):
            #logging.debug(f' {i} : {data_item[0].shape}, {data_item[1]}')
            img = torch.tensor(data=data_item[0].clone().detach(),dtype=torch.float32, device=device)
            img = img.view(-1, 1, 28, 28)
            #label = torch.tensor(data=data_item[1],dtype=torch.float32, device=device)
            y, x, mu, logvar = vvae(img)
            loss = criterion(y, x, mu, logvar)
            if i % 100 == 99:
                print(' {0}, {1}'.format(i, loss.item()))
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()


    print(' Done')