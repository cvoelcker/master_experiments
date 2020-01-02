import torch
from torch import nn
import torch.nn.functional as F

import spatial_monet.util.network_util as net_util

class EncoderNet(nn.Module):
    """
    General parameterized encoding architecture for VAE components
    """

    def __init__(self, image_shape=(32,32), latent_dim=32, input_size=8, **kwargs):
        super().__init__()

        self.latent_dim = latent_dim
        self.image_shape = image_shape

        self.network = nn.Sequential(
            nn.Conv2d(input_size, 32, 3, padding=(1, 1)),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(32, 64, 3, padding=(1, 1)),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(64, 64, 3, padding=(1, 1)),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(64, 128, 3, padding=(1,1)),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, stride=2),
        )
        self.conv_size = 128 * image_shape[0] ** 2) // (2 ** 4) ** 2
        self.mlp = nn.Sequential(
            nn.Linear(self.conv_size, self.conv_size),
            nn.Linear(self.conv_size, 2 * self.latent_dim),
            nn.Linear(2 * self.latent_dim, 2 * self.latent_dim),
            nn.ReLU(inplace=False))
        self.mean_mlp = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim))
        self.sigma_mlp = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim))

    def forward(self, x):
        x = self.network(x)
        x = x.view(-1, self.conv_size)
        x = self.mlp(x)
        mean = self.mean_mlp(x[:, :self.latent_dim])
        sigma = 2 * F.sigmoid(self.sigma_mlp(x[:, self.latent_dim:])) + 1e-5
        return mean, sigma


class DecoderNet(nn.Module):
    """
    General parameterized encoding architecture for VAE components
    """

    def __init__(self, latent_dim=32, image_shape=(32, 32),
                 **kwargs):
        super().__init__()

        self.latent_dim = latent_dim
        self.image_shape = image_shape

        # gave it inverted hourglass shape
        # maybe that helps (random try)
        self.network = nn.Sequential(
            nn.Conv2d(self.latent_dim + 2, 32, 3, padding=(1, 1)),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 64, 5, padding=(2, 2)),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, 3, padding=(1, 1)),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 3, 1),
<<<<<<< Updated upstream
            nn.Sigmoid()
=======
            nn.Sigmoid(),
>>>>>>> Stashed changes
        )

        # coordinate patching trick
        coord_map = net_util.create_coord_buffer(self.image_shape)
        self.register_buffer('coord_map_const', coord_map)

    def forward(self, x):
        # adds coordinate information to z and 
        # produces a tiled representation of z
        z_scaled = x.unsqueeze(-1).unsqueeze(-1)
        z_tiled = z_scaled.repeat(1, 1, self.image_shape[0],
                                  self.image_shape[1])
        coord_map = self.coord_map_const.repeat(x.shape[0], 1, 1, 1)
        inp = torch.cat((z_tiled, coord_map), 1)
        result = self.network(inp)
        return result


class BroadcastVAE(nn.Module):

    def __init__(
            self, bg_sigma=0.1, latent_prior=1.,
            latent_dim=256, image_shape=(256, 256), 
            debug=False, **kwargs):
        super().__init__()
        self.sigma = bg_sigma
        self.latent_dim = latent_dim
        self.image_shape = image_shape
        self.debug = debug
        self.counter = 0

        # networks
        # to encode the masks

        # to encode the images
        self.img_encoder = EncoderNet(
                image_shape=self.image_shape, 
                latent_dim=self.latent_dim,
                input_size=3)
        self.img_decoder = DecoderNet(
                image_shape=self.image_shape,
                latent_dim=self.latent_dim)
        
    def forward(self, x):
        latent_mean, latent_sigma = self.img_encoder(x)
        z, kl = net_util.differentiable_sampling(latent_mean, latent_sigma, 1.)
        recon = self.img_decoder(z)

        recon_loss = net_util.reconstruction_likelihood(x, recon, torch.ones_like(x[:, 0:1, :, :]), self.sigma)

        data_dict = {
            'p_x' : recon,
            'kl_r_loss' : kl,
            'kl_m_loss' : torch.zeros_like(kl),
            'reconstruction': recon.detach(),
        }

        return -torch.sum(recon_loss, (1,2,3)) + torch.sum(kl, 1), data_dict
