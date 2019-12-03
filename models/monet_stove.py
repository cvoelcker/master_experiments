import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Exponential, MultivariateNormal, Normal
from torch.distributions.kl import kl_divergence

from dynamics import Dynamics
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

    def dx_from_state(self, z_sup):
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
        zeros = torch.zeros_like(v[:, 0:1])

        # keep scales and positions from T
        v_full = torch.cat([zeros, v], 1)
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
        zeros = torch.ones_like(v_std[:, 0:1])
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
        dyn_img = z_dyn[..., :self.structured_latent_len]
        std_dyn_img = std_dyn[..., :self.structured_latent_len]
        
        dyn_latent = z_dyn[..., self.structured_latent_len:]
        dyn_std_latent = std_dyn[..., self.structured_latent_len:]

        joined_z_mean = torch.pow(std_img, 2) * dyn_img + torch.pow(std_dyn_img, 2) * z_img
        joined_z_mean = joined_z_mean / (torch.pow(std_dyn_img, 2) + torch.pow(std_img, 2))
        joined_z_std = std_img * std_dyn_img
        joined_z_std = joined_z_std / torch.sqrt(torch.pow(std_dyn_img, 2) + torch.pow(std_img, 2))

        joined_z_full = torch.cat([joined_z_mean, dyn_latent], -1)
        joined_z_std_full = torch.cat([joined_z_std, dyn_std_latent], -1)

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

        T = z_img.shape[1]
        num_obj = self.img_model.num_slots

        # sequence of matched zs
        z_matched = torch.zeros_like(z_img)
        z_std_matched = torch.zeros_like(z_img_std)
        z_matched[:, 0] = z_img[:, 0]
        z_std_matched[:, 0] = z_img_std[:, 0]

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
        return z_matched, z_std_matched

    def match_objects(self, z_img, z_img_std=None, obj_appearances=None):
        """Match objects over sequence.

        No fixed object oder is enforced in SuPAIR. We match object states
        between timesteps by finding the object order which minimizes the
        distance between states at t and t+1.

        Version 1: Worst case complexity O(T*O). Only works for O=num_obj=3.

        Args:
            z_img, z_img_std (torch.Tensor), 2 * (n, T, o, 4): SuPAIR state params.
                Contain id swaps b/c SuPAIR has no notion of time.
            obj_appearances (torch.Tensor), (n, T, o, 3): Appearance information
                may be used to aid matching.
        Returns:
            Permuted version of input arguments.

        """
        # scale to 0, 1 to match appearance information which is already in 0, 1
        z = (z_img + 1)/2
        m_idx = [2, 3]

        if obj_appearances is not None:
            z = torch.cat([z, obj_appearances], -1)
            if self.c.debug_match_appearance:
                # add color channels to comparison
                m_idx += [4, 5, 6]

        if z_img_std is not None:
            z = torch.cat([z, z_img_std], -1)

        T = z.shape[1]
        num_obj = self.img_model.num_slots

        # sequence of matched zs
        z_matched = [z[:, 0]]

        for t in range(1, T):
            # only used to get indices, do not want gradients
            curr = z[:, t, :, m_idx]
            curr = curr.unsqueeze(1).repeat(1, num_obj, 1, 1)
            prev = z_matched[t-1][..., m_idx]
            prev = prev.unsqueeze(2).repeat(1, 1, num_obj, 1)
            # weird bug in pytorch where detaching before unsqueeze would mess
            # with dimensions
            curr, prev = curr.detach(), prev.detach()

            # shape is now (n, o1, o2)
            # o1 is repeat of current, o2 is repeat of previous
            # indexing along o1, we will go through the current values
            # we want to keep o1 fixed, at find minimum along o2
            errors = ((prev - curr)**2).sum(-1)
            _, idx = errors.min(-1)

            # inject some faults for testing
            # idx[-1, :] = torch.LongTensor([0, 0, 0])
            # idx[-2, :] = torch.LongTensor([1, 1, 0])
            # idx[-3, :] = torch.LongTensor([1, 0, 1])
            # idx[1, :] = torch.LongTensor([2, 2, 2])

            """ For an untrained supair, these indices will often not be unique.
                This will likely lead to problems
                Do correction for rows which are affected. How to find them?
                No neighbouring indices can be the same!
                (Here is the reason why curently only 3 objects are supported.)
            """

            faults = torch.prod(idx[:, 1:] != idx[:, :-1], -1)
            faults = faults * (idx[:, 0] != idx[:, 2]).long()
            # at these indexes we have to do greedy matching
            num_faults = (1-faults).sum()

            if num_faults > 0:
                # need to greedily remove faults
                f_errors = errors[faults == 0]
                # sum along current objects
                min_indices = torch.zeros(num_faults, num_obj)
                for obj in range(num_obj):
                    # find first minimum
                    _, f_idx = f_errors.min(-1)
                    # for each seq, 1 column to eliminate
                    # set error values at these indices to large value
                    # s.t. they wont be min again
                    s_idx = f_idx[:, obj]
                    min_indices[:, obj] = s_idx

                    # flatten indices, select correct sequence, then column
                    # (column now selected before row bc transposed)
                    t_idx = torch.arange(s_idx.shape[0]) * num_obj + s_idx
                    tmp = f_errors.permute(0, 2, 1).flatten(end_dim=1)
                    # for all rows
                    tmp[t_idx, :] = 1e12

                    # reshape back to original shape
                    f_errors = tmp.view(f_errors.shape).permute(0, 2, 1)

                # fix faults with new greedily matched
                idx[faults == 0, :] = min_indices.long()

            # select along n, o
            offsets = torch.arange(0, idx.shape[0] * num_obj, num_obj)
            offsets = offsets.unsqueeze(1).repeat(1, num_obj)
            idx_flat = idx + offsets
            idx_flat = idx_flat.flatten()
            z_flat = z[:, t].flatten(end_dim=1)

            match = z_flat[idx_flat].view(z[:, t].shape)
            z_matched += [match]

        z_matched = torch.stack(z_matched, 1)

        # transform back again
        z_img_matched = 2 * z_matched[..., :4] - 1

        if obj_appearances is not None:
            obj_appearances_matched = z_matched[..., 4:7]

        if z_img_std is None and obj_appearances is not None:
            return z_img_matched, obj_appearances_matched

        elif z_img_std is not None and obj_appearances is None:
            z_img_std_matched = z_matched[..., 4:8]
            return z_img_matched, z_img_std_matched, None

        elif z_img_std is not None and obj_appearances is not None:
            z_img_std_matched = z_matched[..., 4+3:8+3]
            return z_img_matched, z_img_std_matched, obj_appearances_matched
        else:
            return z_img_matched

    def _match_objects(self, z_img, z_img_std=None, obj_appearances=None):
        """Match objects over sequence.

        No fixed object oder is enforced in SuPAIR. We match object states
        between timesteps by finding the object order which minimizes the
        distance between states at t and t+1.

        Version 2: Time complexity O(T). Works for all num_obj. BUT: does not
        ensure, that a valid permutation is obtained, i.e., one object may be
        matched multiple times. This has effects largely in the beginning of
        training.

        Args:
            z_img, z_img_std (torch.Tensor), 2 * (n, T, o, 4): SuPAIR state params.
                Contain id swaps b/c SuPAIR has no notion of time.
            obj_appearances (torch.Tensor), (n, T, o, 3): Appearance information
                may be used to aid matching.
        Returns:
            Permuted version of input arguments.

        """
        # scale to 0, 1
        z = (z_img + 1)/2
        m_idx = [2, 3]

        if obj_appearances is not None:
            # colors are already in 0, 1
            z = torch.cat([z, obj_appearances], -1)

            if self.c.debug_match_appearance:
                # add color channels to comparison
                m_idx += [4, 5, 6]

        if z_img_std is not None:
            z = torch.cat([z, z_img_std], -1)

        T = z.shape[1]
        num_obj = self.c.num_obj
        batch_size = z.shape[0]

        # sequence of matched indices
        # list of idx containing object assignments. initialise with any assignment.
        # for each image in sequence
        z_matched = [z[:, 0]]

        for t in range(1, T):

            # only used to get indices, do not want gradients
            curr = z[:, t, :, m_idx]
            prev = z_matched[t-1][..., m_idx]
            # prev changes along rows, curr along columns
            prev = prev.unsqueeze(1).repeat(1, num_obj, 1, 1)
            curr = curr.unsqueeze(2).repeat(1, 1, num_obj, 1)

            # weird bug in pytorch where detaching before unsqueeze would mess
            # with dimensions
            curr, prev = curr.detach(), prev.detach()
            # shape is now (n, o1, o2)
            errors = ((prev - curr)**2).sum(-1)

            # get row-wise minimum, these are column indexes (n, o) for matching
            _, col_idx = errors.min(-2)
            col_idx = col_idx.flatten()
            row_idx = torch.arange(0, col_idx.shape[0])
            # contains for each row (N*num_obj) the col index
            # map to 1d index
            idx = row_idx * num_obj + col_idx

            # from this we obtain permutation matrix by filling zero matrices
            # with ones at min_idxs with regularily increasing rows.
            permutation = torch.zeros(
                (batch_size * num_obj * num_obj),
                dtype=self.c.dtype, device=self.c.device)
            permutation[idx] = 1
            permutation = permutation.view(batch_size, num_obj, num_obj)
            # permute input
            z_perm = torch.matmul(permutation, z[:, t])
            z_matched += [z_perm]

        z_matched = torch.stack(z_matched, 1)

        # transform back again
        z_img_matched = 2 * z_matched[..., :4] - 1

        if obj_appearances is not None:
            obj_appearances_matched = z_matched[..., 4:7]

        if z_img_std is None and obj_appearances is not None:
            return z_img_matched, obj_appearances_matched

        elif z_img_std is not None and obj_appearances is None:
            z_img_std_matched = z_matched[..., 4:8]
            return z_img_matched, z_img_std_matched, None

        elif z_img_std is not None and obj_appearances is not None:
            z_img_std_matched = z_matched[..., 4+3:8+3]
            return z_img_matched, z_img_std_matched, obj_appearances_matched
        else:
            return z_img_matched

    def fix_supair(self, z, z_std=None):
        """Fix misalignments in SuPAIR.

        SuPAIR sometimes glitches. We detect these glitches and replace them by
        averaging the previous and following states, i.e. z_t becomes
        z_t = 0.5 * (z_t-1 + z_t+1).

        Args:
            z, z_std (torch.Tensor), 2 * (n, T, o, 6): SuPAIR states.

         fix weird misalignments in supair

        .. warning : this leaks information from future. in inference model.
            but who says I cant do that? think of it as filtering. who cares.
        """
        if z_std is not None:
            z = torch.cat([z, z_std], -1)

        # find state which has large difference between previous *and* next state
        # this is the weird one

        # get differences between states
        diffs = torch.abs(z[:, 1:, :, :2] - z[:, :-1, :, :2]).detach()

        # for the first one there is no previous
        zeros = torch.zeros([diffs.shape[0], 1, *diffs.shape[2:]],
                            device=self.c.device, dtype=self.c.dtype)
        prev = torch.cat([zeros, diffs], 1)
        # for the last one there is no following
        after = torch.cat([diffs, zeros], 1)

        # get indices of where both diffs are too large
        eps = 0.095
        idxs = (prev > eps) * (after > eps)

        # make table where z_t = (z_t-1 + z_t+1)/2
        smooth = (z[:, :-2] + z[:, 2:]) / 2
        # we now have at t=0: z_2 + z_0, this should go to t=1
        # add zeros in beginning and at end to center
        zeros = torch.zeros([z.shape[0], 1, *z.shape[2:]],
                            device=self.c.device, dtype=self.c.dtype)
        smooth = torch.cat([zeros, smooth, zeros], 1)

        # apply smoothing at idxs
        # right now, idxs is only over positions. we will just assume identical
        # for all dimensions
        idxs = torch.cat(z.shape[-1]//2 * [idxs], -1)

        z_smooth = z
        z_smooth[idxs] = smooth[idxs]

        if z_std is not None:
            z_smooth, z_std_smooth = torch.chunk(z_smooth, 2, dim=-1)
            return z_smooth, z_std_smooth
        else:
            return z_smooth

    def stove_forward(self, x, actions=None, x_color=None):
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
        T = x.shape[1]
        skip = self.skip

        # 1. Obtain SuPAIR states.
        # 1.1 Obtain partial states (sizes and positions).
        # apply supair to all images in sequence. shape (nT, o, 8)
        # supair does not actually have any time_dependence
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
        z_img, z_img_std = self.greedy_matching(z_img, z_img_std)
        assert torch.all(z_img_std > 0)

        # 1.4 Build full states from supair.
        # shape (n, T, o, 2 * (8 + unstructured)), scales, positions and
        # velocities, first full state at T=1 (need 2 imgs)

        z_img_full = self.dx_from_state(z_img)
        z_img_std_full = self.dx_std_from_pos(z_img_std)

        # 2. Dynamics Loop.
        # 2.1 Initialise States.
        # At t=0 we have no dyn, only partial state from supair. see above.
        # At t=1 we have no dyn, however can get full state from supair via
        # supair from t=0. This is used as init for dyn.

        # TODO: interestingly, this probably doesn't have to be changed, since
        # TODO: we push the latents to conform to a 0, 1 prior anyways

        # TODO BIG DECISION: ADDITIONAL PRIOR AND LATENT?
        # TODO: currently, YES
        prior_shape = (*z_img_full[:, skip-1].shape[:-1], self.unstructured_latent_len)
    
        latent_prior = Normal(self.latent_prior_mean, self.latent_prior_std)
        init_z = torch.cat(
            [z_img_full[:, skip-1],
             latent_prior.rsample(prior_shape).squeeze()
             ], -1)
        z = torch.zeros_like(init_z.unsqueeze(1)).repeat(1, T, 1, 1)
        # z_old = (skip-1) * [0] + [init_z] + (T-skip) * [0]
        z[:, skip-1] = init_z

        # Get full time evolution of dyn states. Build step by step.
        # (for sup we already have full time evolution). Initialise z_dyn_std
        # with supair values...
        z_dyn = torch.zeros_like(z)
        z_std_prior = Normal(self.z_std_prior_mean, self.z_std_prior_std)
        # dyn_prior_sample = z_std_prior.rsample(prior_shape).squeeze()
        dyn_prior_sample = z_std_prior.sample(prior_shape).squeeze()
        dyn_std_init = torch.cat([
            z_img_std_full[:, skip-1, :, :],
            dyn_prior_sample
            ], -1)
        
        z_dyn_std = torch.zeros_like(dyn_std_init.unsqueeze(1)).repeat(1, T, 1, 1)
        z_dyn_std[:, skip-1] = dyn_std_init

        z_std = torch.zeros_like(dyn_std_init.unsqueeze(1)).repeat(1, T, 1, 1)
        z_mean = torch.zeros_like(dyn_std_init.unsqueeze(1)).repeat(1, T, 1, 1)
        log_z = torch.zeros_like(z_mean)
        rewards = torch.zeros_like(z[:, :, 1, 1]).squeeze()

        if actions is not None:
            core_actions = actions
        else:
            core_actions = T * [None]

        cur_z = init_z
        
        # 2.2 Loop over sequence and do dynamics prediction.
        for t in range(skip, T):
            # core ignores object sizes
            tmp, reward = self.dyn(cur_z, 0, core_actions[:, t-1])
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

        # 2.3 Stack results from t=skip:T.
        # stack states to (n, T-2, o, 6)
        z_s = z[:, skip:]
        z_dyn_s = z_dyn[:, skip:]
        log_z_s = log_z[:, skip:]
        z_dyn_std_s = z_dyn_std[:, skip:]
        z_std_s = z_std[:, skip:]
        z_mean_s = z_mean[:, skip:]
        
        # 3. Assemble sequence ELBO.
        # 3.1 p(x|z) via SPNs.
        # flatten to (n(T-2)o, depth)
        
        imgs_forward, img_lik_forward, mask_recon_loss = self.img_model.reconstruct_from_latent(
            z_s.flatten(end_dim=1),
            imgs=x[:, skip:].flatten(end_dim=1),
            reconstruct_mask=True)
        # also get lik of initial supair
        imgs_model, img_lik_model, mask_init_recon_loss = self.img_model.reconstruct_from_latent(
            z_img[:, 1:skip].flatten(end_dim=1),
            imgs=x[:, 1:skip].flatten(end_dim=1),
            reconstruct_mask=True)

        # 3.2. Get q(z|x), sample log-likelihoods of inferred z states (n(T-2)).
        log_z_f = log_z_s.sum((-2, -1)).flatten()

        # 3.3 Get p(z_t|z_t-1), generative dynamics distribution.
        trans_lik = self.transition_lik(means=z_dyn_s, results=z_s)
        # sum and flatten shape (n, T-2, o, 6) to (n(T-2))
        trans_lik = trans_lik.sum((-2, -1)).flatten(end_dim=1)
        # 3.4 Finally assemble ELBO.
        elbo = trans_lik + torch.sum(img_lik_forward, [1,2,3]) - log_z_f + self.img_model.gamma * torch.sum(mask_recon_loss, [1,2,3])
        average_elbo = torch.mean(elbo) + torch.mean(img_lik_model) + self.img_model.gamma * torch.mean(mask_init_recon_loss)
        
        prop_dict = {
                'average_elbo': average_elbo.detach().cpu(),
                'trans_lik': trans_lik.detach().cpu(),
                'log_z_f': log_z_f.detach().cpu(),
                'img_lik_forward': torch.sum(img_lik_forward, [1,2,3]).detach().cpu(),
                'elbo': elbo.detach().cpu(),
                'imgs': torch.cat([imgs_forward, imgs_model], 0).detach().cpu(),
                'z_s': z_s.detach().cpu(),
                'img_lik_mean': (torch.mean(img_lik_forward) + torch.mean(img_lik_model)) / 2
                }

        # if ((self.step_counter % self.c.print_every == 0) or
        #         (self.step_counter % self.c.plot_every == 0)):

        #     self.prop_dict['z'] = self.sup.sy_from_quotient(
        #         z_s).detach()
        #     self.prop_dict['z_dyn'] = z_dyn_s.detach()
        #     self.prop_dict['z_img'] = self.sup.sy_from_quotient(
        #         z_img_full[:, skip:]).detach()
        #     self.prop_dict['z_std'] = z_std_s.mean((0, 1, 2)).detach()
        #     self.prop_dict['z_dyn_std'] = torch.cat(
        #         [torch.Tensor([float('nan'), float('nan')], device=self.c.device),
        #          z_dyn_std_s[..., :4].mean((0, 1, 2)).detach()])
        #     self.prop_dict['z_img_std'] = z_img_std_full[:, skip:].mean(
        #         (0, 1, 2)).detach()
        #     self.prop_dict['log_q'] = log_z_f.mean().detach()
        #     self.prop_dict['translik'] = trans_lik.mean().detach()

        #     if self.c.debug and self.c.debug_extend_plots:
        #         self.prop_dict['z_dyn_std_full'] = z_dyn_std_s.detach()
        if self.action_conditioned:
            return average_elbo, prop_dict, rewards[:, skip:]
        else:
            return average_elbo, prop_dict

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
            core_actions = num * [None]

        for t in range(1, num+1):
            tmp, reward = self.dyn(
                    z[t-1], 0, core_actions[:, t-1])
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

            # add back scale
            # z_tmp = torch.cat([scale, z_tmp], -1)

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

    def forward(self, x, step_counter, actions=None, pretrain=False):
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
        self.step_counter = step_counter
        self.img_model.step_counter = step_counter
        self.dyn.step_counter = step_counter
        self.pretrain = pretrain

        if pretrain:
            loss, d = self.img_model(x)
            return loss,d, None
        else:
            return self.stove_forward(x, actions=actions)
