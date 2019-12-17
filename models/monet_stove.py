import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Exponential, MultivariateNormal, Normal
from torch.distributions.kl import kl_divergence

from .dynamics import Dynamics
from spatial_monet.spatial_monet import MaskedAIR


class MONetStove(nn.Module):
    """STOVE: Structured object-aware video prediction model.

    This class combines Masked MONet and the dynamics model.
    """

    def __init__(self, config, dynamics, monet, seed=None):
        """Set up model."""
        super().__init__()
        self.c = config
        # set by calling trainer instance
        self.step_counter = 0

        # export some properties for debugging
        self.prop_dict = {}

        self.img_model = monet
        self.dyn = dynamics
        
        self.cl = self.c.cl
        assert self.cl > 2 * self.img_model.graph_depth, 'Full latent needs to be longer then MONet latent'

        self.full_latent_len = self.cl
        self.structured_latent_len = 2 * self.img_model.graph_depth
        self.unstructured_latent_len = self.cl - 2 * self.img_model.graph_depth

        self.reconstruct_from_z = self.img_model.reconstruct_from_latent

        self.skip = self.c.skip

        # Latent prior for unstructured latent at t=0.
        latent_prior_mean = torch.Tensor([0])
        self.register_buffer('latent_prior_mean', latent_prior_mean)
        latent_prior_std = torch.Tensor([0.01])
        self.register_buffer('latent_prior_std', latent_prior_std)
        z_std_prior_mean = torch.Tensor([0.1])
        self.register_buffer('z_std_prior_mean', z_std_prior_mean)
        z_std_prior_std = torch.Tensor([0.01])
        self.register_buffer('z_std_prior_std', z_std_prior_std)

        self.action_conditioned = config.action_conditioned

        self.dyn_recon_weight = self.c.dyn_recon_weight
        self.pure_img_weight = 0.5

    def dx_from_state(self, z_sup, last_z = None):
        """Get full state by combining supair states.

        The interaction network needs information from at least two previous
        states, since one state encoding does not contain dynamics information

        Args:
            z_sup (torch.Tensor), (n, T, o, img_model.graph_depth): Object
            state means from Image model.

        Returns:
            z_sup_full (torch.Tensor), (n, T, o, 2 * img_model.graph_depth):
            Object state means with pseudo velocities. All zeros at t=0, b/c
            no velocity available.

        """
        # get velocities as differences between positions
        v = z_sup[:, 1:] - z_sup[:, :-1]
        # add zeros to keep time index consistent
        if last_z is None:
            padding = torch.zeros_like(v[:, 0:1])
        else:
            padding = z_sup[:, :1] - last_z[..., :self.img_model.graph_depth]

        # keep scales and positions from T
        v_full = torch.cat([padding, v], 1)
        z_sup_full = torch.cat([z_sup, v_full], -1)

        return z_sup_full

    def dx_std_from_pos(self, z_sup_std):
        """Get std on v from std on positions.

        Args:
            z_sup_std (torch.Tensor), (n, T, o, 4): Object state std from SuPAIR.

        Returns:
            z_sup_std_full (torch.Tensor), (n, T, o, 4): Std with added velocity.

        """
        # Sigma of velocities = sqrt(sigma(x1)**2 + sigma(x2)**2)
        v_std = torch.sqrt(
            z_sup_std[:, 1:]**2 + z_sup_std[:, :-1]**2)
        # This CAN'T be zeros, this is a std
        zeros = torch.ones_like(z_sup_std[:, 0:1])
        v_std_full = torch.cat([zeros, v_std], 1)
        z_sup_std_full = torch.cat([z_sup_std, v_std_full], -1)

        assert torch.all(z_sup_std_full > 0)

        return z_sup_std_full

    def full_state(self, z_dyn, std_dyn, z_img, std_img):
        """Sample full state from dyn and supair predictions at time t.

        Args:
            z_dyn, std_dyn (torch.Tensor), 2 * (n, o, cl//2): Object state means
                and stds from dynamics core. (pos, velo, latent)
            z_img, std_sup (torch.Tensor), 2 * (n, o, 6): Object state means
                and stds from SuPAIR. (size, pos, velo)
        Returns:
            z_s, mean, std (torch.Tensor), 3 * (n, o, cl//2 + 2): Sample of
                full state, SuPAIR and dynamics information combined, means and
                stds of full state distribution.
            log_q (torch.Tensor), (n, o, cl//2 + 2): Log-likelihood of sampled
                state.

        """
        # TODO: currently, there is no more latent just in the dyn model
        # for scales
        z_dyn_shared = z_dyn[..., :self.structured_latent_len]
        std_dyn_shared = std_dyn[..., :self.structured_latent_len]
        
        z_dyn_latent = z_dyn[..., self.structured_latent_len:]
        std_dyn_latent = std_dyn[..., self.structured_latent_len:]

        joined_z_mean = torch.pow(std_img, 2) * z_dyn_shared + torch.pow(std_dyn_shared, 2) * z_img
        joined_z_mean = joined_z_mean / (torch.pow(std_dyn_shared, 2) + torch.pow(std_img, 2))
        joined_z_std = std_img * std_dyn_shared
        joined_z_std = joined_z_std / torch.sqrt(torch.pow(std_dyn_shared, 2) + torch.pow(std_img, 2))

        joined_z_full = torch.cat([joined_z_mean, z_dyn_latent], -1)
        joined_z_std_full = torch.cat([joined_z_std, std_dyn_latent], -1)

        assert torch.all(torch.isfinite(joined_z_std_full))
        assert torch.all(torch.isfinite(joined_z_full))
        dist = Normal(joined_z_full, joined_z_std_full)
        # print(joined_z_std_full.min())

        z_s = dist.rsample()
        assert torch.all(torch.isfinite(z_s))
        # print(torch.min(joined_z_std_full))
        log_q = dist.log_prob(z_s)
        assert torch.all(torch.isfinite(log_q))

        return z_s, log_q, joined_z_full, joined_z_std_full

    def transition_lik(self, means, results):
        """Get generative likelihood of obtained transition.

        The generative dyn part predicts the mean of the distribution over the
        new state z_dyn = 'means'. At inference time, a final state prediction
        z = 'results' is obtained together with information from SuPAIR.
        The generative likelihood of that state is evaluated with distribution
        p(z_t| z_t-1) given by dyn. As in Becker-Ehms, while inference q and p
        of dynamics core share means they do not share stds.

        Args:
            means, results (torch.Tensor), (n, T, o, cl//2): Latent object
                states predicted by generative dynamics core and inferred from
                dynamics model and SuPAIR jointly. States contain
                (pos, velo, latent).

        Returns:
            log_lik (torch.Tensor), (n, T, o, 4): Log-likelihood of results
                under means, i.e. of inferred z under generative model.

        """
        # choose std s.t., if predictions are 'bad', punishment should be high
        assert torch.all(self.dyn.transition_lik_std > 0)
        dist = Normal(means, self.dyn.transition_lik_std)

        log_lik = dist.log_prob(results)

        return log_lik

    def greedy_matching(self, z_img, z_img_std):

        batch = z_img.shape[0]
        T = z_img.shape[1]
        num_obj = self.img_model.num_slots

        # sequence of matched zs
        z_matched = torch.zeros_like(z_img)
        z_std_matched = torch.zeros_like(z_img_std)
        z_matched[:, 0] = z_img[:, 0]
        z_std_matched[:, 0] = z_img_std[:, 0]

        # first matrix has no rotation, because first image is the
        # defining anchor. Input only for consistency with later 
        # functions
        initial = torch.eye(num_obj).unsqueeze(0).repeat(batch, 1, 1).cuda()
        rotations = [initial]

        for t in range(1, T):
            # only used to get indices, do not want gradients
            curr = z_img[:, t, :, :]
            curr = curr.unsqueeze(1).repeat(1, num_obj, 1, 1)
            prev = z_matched[:, t-1, :, :]
            prev = prev.unsqueeze(2).repeat(1, 1, num_obj, 1)
            curr, prev = curr.detach(), prev.detach()

            # shape is now (n, o1, o2)
            # o1 is repeat of current, o2 is repeat of previous
            # indexing along o1, we will go through the current values
            # we want to keep o1 fixed, at find minimum along o2
            errors = ((prev - curr)**2).sum(-1)
            batch, num_obj, _ = errors.shape
            perm_mat = torch.zeros_like(errors)
            errors_max = 2 * torch.max(errors)
            assert torch.all(errors < errors_max)
            for n in range(z_img.shape[2]):
                idx = torch.argmin(errors.view(batch, -1), 1)
                idx_x = idx // num_obj
                idx_y = idx % num_obj
                perm_mat[range(batch), idx_x, idx_y] = 1.
                errors[range(batch), idx_x, :] = errors_max
                errors[range(batch), :, idx_y] = errors_max
            z_matched[:, t] = torch.matmul(perm_mat, z_img[:, t])
            z_std_matched[:, t] = torch.matmul(perm_mat, z_img_std[:, t])
            assert torch.all(z_std_matched[:, t] > 0)
            rotations.append(perm_mat)
        return z_matched, z_std_matched, torch.stack(rotations, 1)

    def encode_sort_img(self, x, last_z = None):
        T = x.shape[1]
        z_img, z_img_std = self.img_model.build_flat_image_representation(
            x.flatten(end_dim=1), return_dists=True)
        # shape (nTo, 4) scales and positions
        # reshape z_img to (n, T, o, img_model.graph_depth)
        nto_shape = (-1, T, self.img_model.num_slots, self.img_model.graph_depth)
        z_img = z_img.view(nto_shape)
        z_img_std = z_img_std.view(nto_shape)
        assert torch.all(z_img_std > 0)

        # 1.2 Find consistent ordering of states.
        # get object embedding from states for latent space
        z_img, z_img_std, permutation = self.greedy_matching(z_img, z_img_std)
        assert torch.all(z_img_std > 0)

        # 1.4 Build full states from supair.
        # shape (n, T, o, 2 * (8 + unstructured)), scales, positions and
        # velocities, first full state at T=1 (need 2 imgs)

        z_img_full = self.dx_from_state(z_img, last_z)
        z_img_std_full = self.dx_std_from_pos(z_img_std)
        return z_img_full, z_img_std_full, permutation

    def infer_dynamics(self, x, actions, z_img_full, z_img_std_full, last_z=None, skip=2):
        T = x.shape[1]
        prior_shape = (*z_img_full[:, skip-1].shape[:-1], self.unstructured_latent_len)

        # if starting new sequene, sample latent from prior
        latent_prior = Normal(self.latent_prior_mean, self.latent_prior_std)
        latent_prior_sample = latent_prior.rsample(prior_shape).squeeze(-1)
        if last_z is None:
            init_z = torch.cat([z_img_full[:, skip-1], latent_prior_sample], -1)
        else:
            init_z = last_z

        z = torch.zeros_like(init_z.unsqueeze(1)).repeat(1, T, 1, 1)
        z_dyn = torch.zeros_like(z)
        z_dyn_std = torch.zeros_like(init_z.unsqueeze(1)).repeat(1, T, 1, 1)
        z_std = torch.zeros_like(init_z.unsqueeze(1)).repeat(1, T, 1, 1)
        z_mean = torch.zeros_like(init_z.unsqueeze(1)).repeat(1, T, 1, 1)
        log_z = torch.zeros_like(z_mean)
        rewards = torch.zeros_like(z[:, :, 1, 1]).squeeze(-1)

        z[:, skip-1] = init_z
        z_std_prior = Normal(self.z_std_prior_mean, self.z_std_prior_std)
        dyn_std_prior_sample = z_std_prior.sample(prior_shape).squeeze(-1)
        dyn_std_init = torch.cat([
            z_img_std_full[:, skip-1, :, :],
            dyn_std_prior_sample
            ], -1)
        z_dyn_std[:, skip-1] = dyn_std_init
        if actions is not None:
            core_actions = actions
        else:
            core_actions = T * [None]
        cur_z = init_z
        
        # 2.2 Loop over sequence and do dynamics prediction.
        for t in range(skip, T):
            # core ignores object sizes
            tmp, reward = self.dyn(cur_z, core_actions[:, t-1])
            # assert not torch.any(torch.isnan(tmp))
            # rewards[:, t] = reward.squeeze()
            z_dyn_tmp, z_dyn_std_tmp = tmp[..., :self.full_latent_len], tmp[..., self.full_latent_len:]
            z_dyn_tmp, z_dyn_std_tmp = self.dyn.constrain_z_dyn(z_dyn_tmp, z_dyn_std_tmp)
            
            cur_z_dyn_std = z_dyn_std_tmp
            # tmp is only a difference on z_dyn[t-1] to z_dyn[t]
            cur_z_dyn = cur_z + z_dyn_tmp

            # obtain full state parameters combining dyn and supair
            # tensors are updated afterwards to prevent inplace assignment issues

            # print(torch.mean(cur_z_dyn[..., :2 * self.img_model.graph_depth] - z_img_full))

            cur_z, cur_log_z, cur_z_mean, cur_z_std = self.full_state(
                    cur_z_dyn, cur_z_dyn_std, z_img_full[:, t], z_img_std_full[:, t])
            
            z[:, t] = cur_z
            log_z[:, t] = cur_log_z
            z_mean[:, t] = cur_z_mean
            z_std[:, t] = cur_z_std
            z_dyn[:, t] = cur_z_dyn
            z_dyn_std[:, t] = cur_z_dyn_std
            
            # assert not torch.any(torch.isnan(log_z))

        return z, log_z, z_dyn, z_dyn_std, z_mean, z_std, rewards

    def stove_forward(self, x, actions=None, x_color=None, last_z = None, mask=None):
        """Forward pass of STOVE.

        n (batch_size), T (sequence length), c (number of channels), w (image
        width), h (image_height)

        Args:
            x (torch.Tensor), (n, T, c, w, h): Sequences of images.
            actions (torch.Tensor), (n, T): Actions for action-conditioned video
                prediction.

        Returns:
            average_elbo (torch.Tensor) (1,): Mean ELBO over sequence.
            self.prop_dict (dict): Dictionary containing performance metrics.
                Used for logging and plotting.

        """
        # mask represents dones form a batch, which masks out invalid transitions
        if mask is None:
            mask = torch.ones_like(x[:, :, 0, 0, 0]).squeeze(-1)
        else:
            mask = 1. - mask
        batch = x.shape[0]
        T = x.shape[1]
        skip = self.skip

        # 1. Obtain SuPAIR states.
        # 1.1 Obtain partial states (sizes and positions).
        # apply supair to all images in sequence. shape (nT, o, 8)
        # supair does not actually have any time_dependence
        z_img, z_img_std, permutation = self.encode_sort_img(x, last_z)

        # 2. Dynamics Loop.
        # 2.1 Initialise States.
        # At t=0 we have no dyn, only partial state from supair. see above.
        # At t=1 we have no dyn, however can get full state from supair via
        # supair from t=0. This is used as init for dyn.

        z, log_z, z_dyn, z_dyn_std, z_mean, z_std, rewards = self.infer_dynamics(x, actions, z_img, z_img_std, last_z, skip)
        z_s = z[:, skip:]
        z_dyn_s = z_dyn[:, skip:]
        log_z_s = log_z[:, skip:]
        z_dyn_std_s = z_dyn_std[:, skip:]
        z_std_s = z_std[:, skip:]
        z_mean_s = z_mean[:, skip:]

        # 3. Assemble sequence ELBO.
        # 3.1 p(x|z) via SPNs.
        # flatten to (n(T-2)o, depth)

        # if permute_back:
        #     raise NotImplementedError()
        
        imgs_forward, img_lik_forward, mask_recon_loss = self.img_model.reconstruct_from_latent(
            z_s.flatten(end_dim=1),
            imgs=x[:, skip:].flatten(end_dim=1),
            reconstruct_mask=True)

        # get reconstructions only from dynamics
        _, img_lik_forward_dyn, mask_recon_loss = self.img_model.reconstruct_from_latent(
            z_dyn_s.flatten(end_dim=1),
            imgs=x[:, skip:].flatten(end_dim=1),
            reconstruct_mask=True)

        # also get lik of initial supair
        imgs_model, img_lik_model, mask_init_recon_loss = self.img_model.reconstruct_from_latent(
            z_img[:, :skip].flatten(end_dim=1),
            imgs=x[:, :skip].flatten(end_dim=1),
            reconstruct_mask=True)

        # 3.2. Get q(z|x), sample log-likelihoods of inferred z states (n(T-2)).
        log_z_f_masked = (mask[:, self.skip:] * log_z_s.sum((-2, -1)))

        # 3.3 Get p(z_t|z_t-1), generative dynamics distribution.
        trans_lik = self.transition_lik(means=z_dyn_s, results=z_s)

        trans_lik_masked = (mask[:, self.skip:] * trans_lik.sum((-2, -1)))
        img_lik_forward_masked = (mask[:, self.skip:] * img_lik_forward.sum((-3, -2, -1)).view(batch, T-skip))
        img_lik_model_masked = (mask[:, :self.skip] * img_lik_model.sum((-3, -2, -1)).view(batch, skip))
        img_lik_forward_dyn_masked = (mask[:, self.skip:] * img_lik_forward_dyn.sum((-3, -2, -1)).view(batch, T-skip))
        mask_recon_loss_masked = (mask[:, self.skip:] * mask_recon_loss.sum((-3, -2, -1)).view(batch, T-skip))

        # 3.4 Finally assemble ELBO.
        elbo = trans_lik_masked + img_lik_forward_masked - log_z_f_masked
        augmented_elbo = (T-skip)/T * torch.mean(elbo) + \
                       self.pure_img_weight * torch.mean(img_lik_model_masked) + \
                       self.dyn_recon_weight * torch.mean(img_lik_forward_dyn_masked) + \
                       self.img_model.gamma * torch.mean(mask_init_recon_loss) + \
                       self.img_model.gamma * mask_recon_loss_masked
        
        prop_dict = {
                'average_elbo': augmented_elbo.detach().cpu(),
                'trans_lik': trans_lik.detach().cpu(),
                'log_z_f': log_z_f_masked.detach().cpu(),
                'img_lik_forward': torch.sum(img_lik_forward, [1,2,3]).detach().cpu(),
                'elbo': elbo.detach().cpu(),
                'z_s': z_s.cpu().detach(),
                'img_lik_mean': torch.mean(torch.cat((img_lik_forward_masked, img_lik_model_masked), 1)).detach().cpu(),
                'imgs': img_lik_model.cpu().detach(),
                }

        if self.action_conditioned:
            return augmented_elbo, prop_dict, rewards[:, skip:]
        else:
            return augmented_elbo, prop_dict

    def rollout(self, z_last, num=None, sample=False, return_std=False,
                actions=None, appearance=None, return_imgs=False):
        """Rollout a given state using the dynamics model.

        Args:
            z_last (torch.Tensor), (n, o, cl//2 + 2): Object states as produced,
                e.g., by prop_dict['z'] in vin_forward(). Careful z_last
                contains [sx, sy] and not [sx, sy/sx].
            num (int): Number of states to roll out.
            sample (bool): Sample from distribution from dynamics core instead
                of predicting the mean.
            return_std (bool): Return std of distribution from dynamics model.
            actions torch.Tensor, (n, T): Actions to apply to dynamics model,
                affecting the rollout.
            appearance torch.Tensor, (n, T, o, 3): Appearance information, as
                aid for dynamics model. Assumed constant during rollout.

        Returns:
            z_pred: (n, num, o, cl//2 + 2) Predictions over future states.

        """
        cl = self.cl
        if num is None:
            num = self.c.num_rollout

        z = [z_last]
        # keep scale constant during rollout
        rewards = [0.] * num
        # need last z_dyn_std
        if sample or return_std:
            log_qs = []
        # std for first state not given
        if return_std:
            z_dyn_stds = []
        if actions is not None:
            core_actions = actions
        else:
            core_actions = num * [0.]

        for t in range(1, num+1):
            tmp, reward = self.dyn(
                    z[t-1], core_actions[:, t-1])
            z_tmp, z_dyn_std_tmp = tmp[..., :self.full_latent_len], tmp[..., self.full_latent_len:]
            rewards[t-1] = reward

            z_tmp, z_dyn_std = self.dyn.constrain_z_dyn(z_tmp, z_dyn_std_tmp)
            
            z_dyn = z[t-1] + z_tmp

            if sample or return_std:
                z_dyn_stds.append(z_dyn_std)
                if sample:
                    dist = Normal(z_dyn, z_dyn_std)
                    z_tmp = dist.rsample()
                    log_qs.append(dist.log_prob(z_dyn))

            z.append(z_dyn)

        if self.c.action_conditioned:
            rewards = torch.stack(rewards, 1)
        else:
            rewards = torch.Tensor(rewards)

        # first state was given, not part of rollout
        z_full = torch.stack(z, 1)
        if sample:
            return z_full, torch.stack(log_qs, 1), rewards
        if return_imgs:
            imgs = self.img_model.reconstruct_from_latent(z_full.flatten(end_dim=1), imgs=None, reconstruct_mask=True)
            imgs = imgs.reshape(z_last.shape[0], num+1, 3, *self.img_model.image_shape)
            return z_full, imgs, rewards
        if return_std:
            z_dyn_stds = torch.stack(z_dyn_stds, 1)
            return z_full, z_dyn_stds.detach(), rewards
        else:
            return z_full, rewards

    def forward(self, x, actions=None, rewards=None, mask=None, pretrain=False):
        """Forward function.

        Can be used to train (action-conditioned) video prediction or
        SuPAIR only without any dynamics model.

        Args:
            x (torch.Tensor), (n, T, o, 3, w, h): Color images..
            step_counter (int): Current training progress.
            actions (torch.Tensor) (n ,T): Actions from env.
            pretrain (bool): Switch for SuPAIR-only training.

        Returns:
            Whatever the respective forwards return.

        """
        self.pretrain = pretrain

        if pretrain:
            loss, d = self.img_model(x.flatten(end_dim=1))
            return -1. * loss, d, None
        else:
            return self.stove_forward(x, actions=actions, mask=mask)

    def infer_latent(self, obs, actions):
        z_img, z_img_std, _ = self.encode_sort_img(obs)
        z_tuple = self.infer_dynamics(obs, actions, z_img, z_img_std, skip=2)
        return z_tuple[0]

    def update_latent(self, obs, actions, latent):
        # if a sequence should be appended with more latents
        z_img, z_img_std, _ = self.encode_sort_img(obs, latent)
        z, _, _, _, _, _, _ = self.infer_dynamics(obs, actions, z_img, z_img_std, last_z=latent, skip=1)
        return z[:, -1]
