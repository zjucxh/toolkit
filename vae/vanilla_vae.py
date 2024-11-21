import torch
import logging
from torch import nn
from torch.nn import functional as F
from typing import List, Callable, Union, Any, TypeVar, Tuple


class VanillaVAE(nn.Module):
    def __init__(self, in_channels:int, width:int, height:int, dim_latent:int, hidden_dims:List=None) -> None:
        super().__init__()
        self.dim_latent = dim_latent
        self.in_channels = in_channels
        self.width = width
        self.height = height
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128 ]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(nn.Conv2d(in_channels=in_channels,out_channels=h_dim, 
                                        kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU()
            ))
            in_channels=h_dim
        
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4*4, self.dim_latent) # TODO: tackling magic number
        self.fc_var = nn.Linear(hidden_dims[-1] *4*4, self.dim_latent) # TODO: tackling magic number

        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(dim_latent, hidden_dims[-1]*4*4) # TODO: tackling magic number
        hidden_dims.reverse()
        for i in range(len(hidden_dims)-1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i+1],kernel_size=3, stride=2, padding=1,output_padding=0),
                    nn.BatchNorm2d(hidden_dims[i+1]),
                    nn.LeakyReLU()
                )
            )
        self.decoder = nn.Sequential(*modules)
        self.conv_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden_dims[-1],out_channels=hidden_dims[-1], 
                               kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU()
        )
        self.linear_layer = nn.Sequential(
            nn.Linear(21632, 784),
            nn.ReLU()
        )



    def encode(self, input:torch.tensor):
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var 
    
    def reparameterize(self, mu:torch.tensor, logvar:torch.tensor)->torch.tensor:
        """
        Paramaterization trick to sample from N(mu, var) from N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def decode(self, z:torch.tensor):
        y = self.decoder_input(z)
        y = y.view(y.shape[0], -1, 4,4)
        y = self.decoder(y)
        y = self.conv_layer(y)
        y = y.view(y.shape[0],-1)
        y = self.linear_layer(y)
        y = y.view(y.shape[0], 1, 28, 28)
        return y
    
    def forward(self, input:torch.Tensor)->List[torch.tensor]:
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        return [self.decode(z), input, mu, logvar]
    
    
    def loss_func(self, output, input, mu, logvar):
        recon_loss = F.mse_loss(input, output)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
        #logging.debug(' kld loss in loss function2 : {0}'.format(kld_loss))
        return recon_loss + kld_loss



if __name__=='__main__':
    logging.basicConfig(level=logging.DEBUG)
    # create random datasample
    x = torch.randn(size=(64,1,28,28),dtype=torch.float32)
    vvae = VanillaVAE(in_channels=1, width=28, height=28, dim_latent=72,hidden_dims=[32,64,128])
    y, input, mu, logvar = vvae(x)
    loss = vvae.loss_function(y, input, mu, logvar)
    loss2 = vvae.loss_func(y, input, mu, logvar)
    logging.debug(' loss : {0}'.format(loss))
    logging.debug(' loss2 : {0}'.format(loss2))
    print('Done')