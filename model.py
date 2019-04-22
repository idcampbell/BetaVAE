import json
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from torchvision import transforms
import torchvision
import utils

# Load in the parameters which are relevant to the training process.
with open('params.json') as json_file:  
    params = json.load(json_file)
    z_dim = params["z_dim"]
    beta = params["beta"]


class BetaVAE(nn.Module):
    
    def __init__(self):
        super(BetaVAE, self).__init__()
        
        self.z_dim = z_dim
        self.beta = beta
        
        # Define the sequence of operations in the encoder.
        self.encoder = nn.Sequential(
            self._conv(3, 32),       # BATCH, 32, 32, 32
            self._conv(32, 64),      # BATCH, 64, 16, 16
            self._conv(64, 128),     # BATCH, 128, 8, 8
            self._conv(128, 256),    # BATCH, 256, 4, 4
        )
        
        # Mean and variance layers.
        self.fc_mu = nn.Linear(4096, self.z_dim)
        self.fc_var = nn.Linear(4096, self.z_dim)
        
        # Rescale the data to a larger feature set.
        self.fc_z = nn.Linear(self.z_dim, 4096)
        
        # Define the sequence of operations in the decoder.
        self.decoder = nn.Sequential(
            self._deconv(256, 128),  # BATCH, 128, 8, 8
            self._deconv(128, 64),   # BATCH, 64, 16, 16
            self._deconv(64, 32),    # BATCH, 32, 32, 32
            self._deconv(32, 3),     # BATCH, 3, 64, 64
            nn.Sigmoid()
        )

        # Initialize the network weights using Kaiming initialization (https://arxiv.org/abs/1502.01852)
        self.weight_init()
        
    def encode(self, x):
        x = self.encoder(x)
        x = x.view(-1, 4096)
        return self.fc_mu(x), self.fc_var(x)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, z):
        z = self.fc_z(z)
        z = z.view(-1, 256, 4, 4)
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
    def _conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
    
    def _deconv(self, in_channels, out_channels, padding=1):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def loss(self, x_recon, x, mu, logvar):
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return (recon_loss + self.beta * KLD) / x.shape[0]
    
    def save_model(self, file_path, num_to_keep=1):
        utils.save(self, file_path, num_to_keep)
        
    def load_model(self, file_path):
        utils.restore(self, file_path)
        
    def load_last_model(self, dir_path):
        return utils.restore_latest(self, dir_path)
    
    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)
                
    def kaiming_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            init.kaiming_normal(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif isinstance(m, (nn.BatchNorm2d)):
            m.weight.data.fill_(1)
            if m.bias is not None:
                m.bias.data.fill_(0)
