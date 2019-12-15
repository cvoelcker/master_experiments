import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as dists
from torch.distributions.kl import kl_divergence
import numpy as np


class BaseNetwork(nn.Module):
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


def create_linear_network(input_dim, output_dim, hidden_units=[256, 256],
                          hidden_activation=nn.ReLU(), output_activation=None,):
    model = []
    units = input_dim
    for next_units in hidden_units:
        model.append(nn.Linear(units, next_units))
        model.append(hidden_activation)
        units = next_units

    model.append(nn.Linear(units, output_dim))
    if output_activation is not None:
        model.append(output_activation)

    return nn.Sequential(*model)


def calc_kl_divergence(p_list, q_list):
    assert len(p_list) == len(q_list)

    kld = 0.0
    for i in range(len(p_list)):
        # (N, L) shaped array of kl divergences.
        _kld = kl_divergence(p_list[i], q_list[i])
        # Average along batches, sum along sequences and elements.
        kld += _kld.mean(dim=0).sum()

    return kld


class SLACModel(nn.Module):

    def __init__(self, observation_shape, action_shape, feature_dim=64,
                 latent1_dim=32, latent2_dim=32, hidden_units=[64, 64],
                 leaky_slope=0.2):
        super().__init__()
        self.latent = LatentNetwork(observation_shape, action_shape, feature_dim=feature_dim,
                 latent1_dim=latent1_dim, latent2_dim=latent2_dim, hidden_units=hidden_units,
                 leaky_slope=leaky_slope)
        self.cl = feature_dim

    def forward(self, images_seq, actions_seq, rewards_seq,
                         dones_seq):
        features_seq = self.latent.encoder(images_seq)

        # Sample from posterior dynamics.
        (latent1_post_samples, latent2_post_samples),\
            (latent1_post_dists, latent2_post_dists) =\
            self.latent.sample_posterior(features_seq, actions_seq)
        # Sample from prior dynamics.
        (latent1_pri_samples, latent2_pri_samples),\
            (latent1_pri_dists, latent2_pri_dists) =\
            self.latent.sample_prior(actions_seq)

        # KL divergence loss.
        kld_loss = calc_kl_divergence(latent1_post_dists, latent1_pri_dists)

        # Log likelihood loss of generated observations.
        images_seq_dists = self.latent.decoder(
            [latent1_post_samples, latent2_post_samples])
        log_likelihood_loss = images_seq_dists.log_prob(
            images_seq).mean(dim=0).sum()

        # Log likelihood loss of genarated rewards.
        rewards_seq_dists = self.latent.reward_predictor([
            latent1_post_samples,
            latent2_post_samples,
            actions_seq, latent1_post_samples,
            latent2_post_samples])
        reward_log_likelihoods =\
            rewards_seq_dists.log_prob(rewards_seq.unsqueeze(-1)) * (1.0 - dones_seq).unsqueeze(-1)
        reward_log_likelihood_loss = reward_log_likelihoods.mean(dim=0).sum()

        latent_loss =\
            kld_loss - log_likelihood_loss - reward_log_likelihood_loss

        return -1 * latent_loss, None, None
    
    def infer_latent(self, images_seq, actions_seq):
        features_seq = self.latent.encoder(images_seq)

        # from IPython.core.debugger import Tracer
        # Tracer()()
        # Sample from posterior dynamics.
        (latent1_post_samples, latent2_post_samples),\
            (latent1_post_dists, latent2_post_dists) =\
            self.latent.sample_posterior(features_seq, actions_seq)
        latents = torch.cat((latent1_post_samples, latent2_post_samples), -1)
        return latents

    def update_latent(self, new_image, new_action, last_latent):
        features = self.latent.encoder(new_image)
        (latent1_post_samples, latent2_post_samples),\
            (latent1_post_dists, latent2_post_dists) =\
            self.latent.sample_posterior(features, new_action)
        latents = torch.cat((latent1_post_samples, latent2_post_samples), -1)
        return latents[:, -1]


class Gaussian(BaseNetwork):

    def __init__(self, input_dim, output_dim, hidden_units=[256, 256],
                 std=None, leaky_slope=0.2):
        super(Gaussian, self).__init__()
        self.net = create_linear_network(
            input_dim, 2*output_dim if std is None else output_dim,
            hidden_units=hidden_units,
            hidden_activation=nn.LeakyReLU(leaky_slope),)

        self.std = std

    def forward(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            x = torch.cat(x, dim=-1)

        x = self.net(x)
        if self.std:
            mean = x
            std = torch.ones_like(mean) * self.std
        else:
            mean, std = torch.chunk(x, 2, dim=-1)
            std = F.softplus(std) + 1e-5

        return dists.Normal(loc=mean, scale=std)


class ConstantGaussian(BaseNetwork):

    def __init__(self, output_dim, std=1.0):
        super(ConstantGaussian, self).__init__()
        self.output_dim = output_dim
        self.std = std

    def forward(self, x):
        mean = torch.zeros((x.size(0), self.output_dim)).to(x)
        std = torch.ones((x.size(0), self.output_dim)).to(x) * self.std
        return dists.Normal(loc=mean, scale=std)


class Decoder(BaseNetwork):

    def __init__(self, input_dim=288, output_dim=3, std=1.0, leaky_slope=0.2):
        super(Decoder, self).__init__()
        self.std = std

        self.net = nn.Sequential(
            # (32+256, 1, 1) -> (256, 4, 4)
            nn.ConvTranspose2d(input_dim, 256, 4),
            nn.LeakyReLU(leaky_slope),
            # (256, 4, 4) -> (128, 8, 8)
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
            nn.LeakyReLU(leaky_slope),
            # (128, 8, 8) -> (64, 16, 16)
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.LeakyReLU(leaky_slope),
            # (64, 16, 16) -> (32, 32, 32)
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.LeakyReLU(leaky_slope),
            # (32, 32, 32) -> (3, 64, 64)
            nn.ConvTranspose2d(32, 3, 5, 2, 2, 1),
            nn.LeakyReLU(leaky_slope)
        )

    def forward(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            x = torch.cat(x,  dim=-1)

        num_batches, num_sequences, latent_dim = x.size()
        x = x.view(num_batches * num_sequences, latent_dim, 1, 1)
        x = self.net(x)
        _, C, W, H = x.size()
        x = x.view(num_batches, num_sequences, C, W, H)
        return dists.Normal(loc=x, scale=torch.ones_like(x) * self.std)


class Encoder(BaseNetwork):

    def __init__(self, input_dim=3, output_dim=256, leaky_slope=0.2):
        super(Encoder, self).__init__()

        self.net = nn.Sequential(
            # (3, 64, 64) -> (32, 32, 32)
            nn.Conv2d(input_dim, 32, 5, 2, 2),
            nn.LeakyReLU(leaky_slope),
            # (32, 32, 32) -> (64, 16, 16)
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.LeakyReLU(leaky_slope),
            # (64, 16, 16) -> (128, 8, 8)
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.LeakyReLU(leaky_slope),
            # (128, 8, 8) -> (256, 4, 4)
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.LeakyReLU(leaky_slope),
            # (256, 4, 4) -> (256, 1, 1)
            nn.Conv2d(256, output_dim, 4),
            nn.LeakyReLU(leaky_slope)
        )

    def forward(self, x):
        num_batches, num_sequences, C, H, W = x.size()
        x = x.view(num_batches * num_sequences, C, H, W)
        x = self.net(x)
        x = x.view(num_batches, num_sequences, -1)

        return x


class LatentNetwork(BaseNetwork):

    def __init__(self, observation_shape, action_shape, feature_dim=64,
                 latent1_dim=32, latent2_dim=32, hidden_units=[64, 64],
                 leaky_slope=0.2):
        super(LatentNetwork, self).__init__()
        # NOTE: We encode x as the feature vector to share convolutional
        # part of the network with the policy.

        self.latent1_dim = latent1_dim
        self.latent2_dim = latent2_dim

        # p(z1(0)) = N(0, I)
        self.latent1_init_prior = ConstantGaussian(latent1_dim)
        # p(z2(0) | z1(0))
        self.latent2_init_prior = Gaussian(
            latent1_dim, latent2_dim, hidden_units, leaky_slope=leaky_slope)
        # p(z1(t+1) | z2(t), a(t))
        self.latent1_prior = Gaussian(
            latent2_dim + action_shape, latent1_dim, hidden_units,
            leaky_slope=leaky_slope)
        # p(z2(t+1) | z1(t+1), z2(t), a(t))
        self.latent2_prior = Gaussian(
            latent1_dim + latent2_dim + action_shape, latent2_dim,
            hidden_units, leaky_slope=leaky_slope)

        # q(z1(0) | feat(0))
        self.latent1_init_posterior = Gaussian(
            feature_dim, latent1_dim, hidden_units, leaky_slope=leaky_slope)
        # q(z2(0) | z1(0)) = p(z2(0) | z1(0))
        self.latent2_init_posterior = self.latent2_init_prior
        # q(z1(t+1) | feat(t+1), z2(t), a(t))
        self.latent1_posterior = Gaussian(
            feature_dim + latent2_dim + action_shape, latent1_dim,
            hidden_units, leaky_slope=leaky_slope)
        # q(z2(t+1) | z1(t+1), z2(t), a(t)) = p(z2(t+1) | z1(t+1), z2(t), a(t))
        self.latent2_posterior = self.latent2_prior

        # p(r(t) | z1(t), z2(t), a(t), z1(t+1), z2(t+1))
        self.reward_predictor = Gaussian(
            2 * latent1_dim + 2 * latent2_dim + action_shape,
            1, hidden_units, leaky_slope=leaky_slope)

        # feat(t) = x(t) : This encoding is performed deterministically.
        self.encoder = Encoder(
            observation_shape[2], feature_dim, leaky_slope=leaky_slope)
        # p(x(t) | z1(t), z2(t))
        self.decoder = Decoder(
            latent1_dim + latent2_dim, observation_shape[2],
            std=np.sqrt(0.1), leaky_slope=leaky_slope)

    def sample_prior(self, actions_seq, init_features=None):
        ''' Sample from prior dynamics (with conditionning on the initial frames).
        Args:
            actions_seq   : (N, S, *action_shape) tensor of action sequences.
            init_features : (N, *) tensor of initial frames or None.
        Returns:
            latent1_samples : (N, S+1, L1) tensor of sampled latent vectors.
            latent2_samples : (N, S+1, L2) tensor of sampled latent vectors.
            latent1_dists   : (S+1) length list of (N, L1) distributions.
            latent2_dists   : (S+1) length list of (N, L2) distributions.
        '''
        num_sequences = actions_seq.size(1)
        actions_seq = torch.transpose(actions_seq, 0, 1)

        latent1_samples = []
        latent2_samples = []
        latent1_dists = []
        latent2_dists = []

        for t in range(num_sequences):
            if t == 0:
                # Condition on initial frames.
                if init_features is not None:
                    # q(z1(0) | feat(0))
                    latent1_dist = self.latent1_init_posterior(init_features)
                    latent1_sample = latent1_dist.rsample()
                    # q(z2(0) | z1(0))
                    latent2_dist = self.latent2_init_posterior(latent1_sample)
                    latent2_sample = latent2_dist.rsample()

                # Not conditionning.
                else:
                    # p(z1(0)) = N(0, I)
                    latent1_dist = self.latent1_init_prior(actions_seq[t])
                    latent1_sample = latent1_dist.rsample()
                    # p(z2(0) | z1(0))
                    latent2_dist = self.latent2_init_prior(latent1_sample)
                    latent2_sample = latent2_dist.rsample()

            else:
                # p(z1(t) | z2(t-1), a(t-1))
                latent1_dist = self.latent1_prior(
                    [latent2_samples[t-1], actions_seq[t-1]])
                latent1_sample = latent1_dist.rsample()
                # p(z2(t) | z1(t), z2(t-1), a(t-1))
                latent2_dist = self.latent2_prior(
                    [latent1_sample, latent2_samples[t-1], actions_seq[t-1]])
                latent2_sample = latent2_dist.rsample()

            latent1_samples.append(latent1_sample)
            latent2_samples.append(latent2_sample)
            latent1_dists.append(latent1_dist)
            latent2_dists.append(latent2_dist)

        latent1_samples = torch.stack(latent1_samples, dim=1)
        latent2_samples = torch.stack(latent2_samples, dim=1)

        return (latent1_samples, latent2_samples),\
            (latent1_dists, latent2_dists)

    def sample_posterior(self, features_seq, actions_seq, last_latent=None):
        ''' Sample from posterior dynamics.
        Args:
            features_seq : (N, S+1, 256) tensor of feature sequenses.
            actions_seq  : (N, S, *action_space) tensor of action sequenses.
        Returns:
            latent1_samples : (N, S+1, L1) tensor of sampled latent vectors.
            latent2_samples : (N, S+1, L2) tensor of sampled latent vectors.
            latent1_dists   : (S+1) length list of (N, L1) distributions.
            latent2_dists   : (S+1) length list of (N, L2) distributions.
        '''
        num_sequences = actions_seq.size(1)
        features_seq = torch.transpose(features_seq, 0, 1)
        actions_seq = torch.transpose(actions_seq, 0, 1)

        latent1_samples = []
        latent2_samples = []
        latent1_dists = []
        latent2_dists = []

        for t in range(num_sequences):
            if t == 0 and last_latent is None:
                # q(z1(0) | feat(0))
                latent1_dist = self.latent1_init_posterior(features_seq[t])
                latent1_sample = latent1_dist.rsample()
                # q(z2(0) | z1(0))
                latent2_dist = self.latent2_init_posterior(latent1_sample)
                latent2_sample = latent2_dist.rsample()
            elif t==0 and last_latent is not None:
                latent1_dist = self.latent1_init_posterior(features_seq[t])
                latent1_sample = last_latent[..., :self.latent1_dim]
                latent2_dist = self.latent2_init_posterior(latent1_sample)
                latent2_sample = last_latent[..., self.latent1_dim:]
            else:
                # q(z1(t) | feat(t), z2(t-1), a(t-1))
                latent1_dist = self.latent1_posterior(
                    [features_seq[t], latent2_samples[t-1], actions_seq[t-1]])
                latent1_sample = latent1_dist.rsample()
                # q(z2(t) | z1(t), z2(t-1), a(t-1))
                latent2_dist = self.latent2_posterior(
                    [latent1_sample, latent2_samples[t-1], actions_seq[t-1]])
                latent2_sample = latent2_dist.rsample()

            latent1_samples.append(latent1_sample)
            latent2_samples.append(latent2_sample)
            latent1_dists.append(latent1_dist)
            latent2_dists.append(latent2_dist)

        latent1_samples = torch.stack(latent1_samples, dim=1)
        latent2_samples = torch.stack(latent2_samples, dim=1)

        return (latent1_samples, latent2_samples),\
            (latent1_dists, latent2_dists)

