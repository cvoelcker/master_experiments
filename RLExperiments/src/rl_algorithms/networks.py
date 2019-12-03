import torch
from torch import nn
from torch.nn import functional as F

from spatial_monet import spatial_monet
from spatial_monet.util import experiment_config

from math import floor


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def calc_output_max_pool(h, w, k, s, p, d=0):
    h = floor((h + 2 * p - d * (k - 1) - 1) / s + 1)
    w = floor((w + 2 * p - d * (k - 1) - 1) / s + 1)
    return h, w


class ImageInputNetwork(nn.Module):

    def __init__(self, image_shape, z_dim, num_blocks, dropout=False,
                 subsampling=True, embedding=128):
        """
        Module to provide initial downsampling of image input

        :param image_shape: int tuple of the image shape
        :param z_dim: number of latent variables in the compression
        :param num_blocks: number of convolution max pool layers
        :param dropout: add dropout after initial convolution
        """
        super().__init__()

        self.image_shape = image_shape
        self.z_dim = z_dim
        self.num_blocks = num_blocks

        self.layers = nn.ModuleList()

        channels = self.image_shape[2]
        shape_x = self.image_shape[0]
        shape_y = self.image_shape[1]

        if subsampling:
            assert shape_x % (2 ** num_blocks) == 0, \
                'Image is not evenly divisible by max pooling layer'
            assert shape_y % (2 ** num_blocks) == 0, \
                'Image is not evenly divisible by max pooling layer'

            for i in range(num_blocks):
                self.layers.append(
                    nn.Conv2d(channels, channels * 4, 3, padding=1))
                self.layers.append(nn.ReLU())
                self.layers.append(nn.MaxPool2d(2, 2))

                channels = channels * 4
                shape_x = int(shape_x / 2)
                shape_y = int(shape_y / 2)

            self.linear_input = channels * shape_x * shape_y
            self.linear = nn.Linear(channels * shape_x * shape_y, z_dim)

        else:
            block_shape = [8, 4, 3]
            block_strides = [4, 2, 1]
            filters = [16, 32, 64]
            for i in range(num_blocks):
                self.layers.append(
                    nn.Conv2d(channels, filters[i], block_shape[i],
                              stride=block_strides[i]))
                self.layers.append(nn.ReLU())

                channels = filters[i]
                # calculation taken from https://pytorch.org/docs/stable
                # nn.html#torch.nn.Conv2d
                shape_x = int(((shape_x - (block_shape[i] - 1) - 1) /
                               block_strides[i]) + 1)
                shape_y = int(((shape_y - (block_shape[i] - 1) - 1) /
                               block_strides[i]) + 1)

            self.linear_input = int(channels * shape_x * shape_y)
            self.linear = nn.Linear(self.linear_input, embedding)

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        x = x.view(-1, self.linear_input)
        return self.linear(x)


class ActorNet(nn.Module):

    def __init__(self, input_dim, action_space, hiddens=[]):
        """
        Simple feed forward network which outputs a categorical distribution
        :param input_dim: number of inputs
        :param action_space: number of outputs
        :param hiddens: list of int containing
        """
        super().__init__()
        self.input_dim = input_dim
        self.action_space = action_space

        self.layers = nn.ModuleList()
        self.hidden = hiddens.copy()

        self.hidden.append(action_space)
        inp = input_dim

        for h in self.hidden:
            self.layers.append(nn.Linear(inp, h))
            self.layers.append(nn.ReLU())
            inp = h
        self.layers.append(nn.Softmax(dim=-1))

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


class ImageActor(nn.Module):
    def __init__(self,
                 image_shape,
                 z_dim,
                 num_blocks,
                 action_space,
                 hiddens=[],
                 dropout=False,
                 subsampling=True):
        """
        Combines image preprocessing and actor network into one for usability
        :param image_shape:
        :param z_dim:
        :param num_blocks:
        :param action_space:
        :param hiddens:
        :param dropout:
        """
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            ImageInputNetwork(image_shape, z_dim, num_blocks, dropout,
                              subsampling))
        self.layers.append(ActorNet(action_space, z_dim, hiddens))

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


class CriticNet(nn.Module):

    def __init__(self, input_dim, hidden=[]):
        super().__init__()
        self.input_dim = input_dim

        self.layers = nn.ModuleList()
        self.hidden = hidden.copy()

        inp = input_dim

        for h in self.hidden:
            self.layers.append(nn.Linear(inp, h))
            self.layers.append(nn.ReLU())
            inp = h
        self.layers.append(nn.Linear(inp, 1))

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


class ActorCriticImageNet(nn.Module):
    def __init__(self,
                 image_shape,
                 z_dim,
                 num_blocks,
                 action_space,
                 hiddens=[],
                 dropout=False,
                 subsampling=True):
        """
        Combines image preprocessing and actor network into one for usability
        :param image_shape:
        :param z_dim:
        :param num_blocks:
        :param action_space:
        :param hiddens:
        :param dropout:
        """
        super().__init__()
        self.image_shape = image_shape
        self.layers = nn.ModuleList()
        self.layers.append(
            ImageInputNetwork(image_shape, z_dim, num_blocks, dropout,
                              subsampling))
        self.layers.append(nn.Sequential(
            nn.Linear(128, z_dim),
            nn.ReLU()
        ))
        self.layers.append(nn.Sequential(nn.Linear(z_dim, action_space.n),
                                         nn.Softmax(1)))
        self.layers.append(nn.Linear(z_dim, 1))

    def forward(self, x):
        img_embedding = self.layers[0](x)
        img_embedding = self.layers[1](img_embedding)
        action = self.layers[2](img_embedding)
        value = self.layers[3](img_embedding)
        return action, value

    def image_input_shape(self):
        return self.image_shape[2], self.image_shape[0], self.image_shape[1]


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)


class FCNetwork(nn.Module):
    def __init__(self, image_shape, action_space, hidden_size=512):
        super().__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.image_shape = image_shape
        self.image_size = image_shape[0] * image_shape[1] * image_shape[2]
        self.layers = nn.Sequential(
            Flatten(),
            nn.Linear(self.image_size, hidden_size),
            nn.ReLU(),
            # init_(nn.Linear(hidden_size, hidden_size // 2)),
            # nn.ReLU())
        )
        self.policy_logits = nn.Linear(hidden_size, action_space.n)
        self.critic_linear = nn.Linear(hidden_size, 1)

    def forward(self, inputs):
        x = self.layers(inputs / 255.)
        return self.policy_logits(x), self.critic_linear(x)

    def image_input_shape(self):
        return self.image_shape[2], self.image_shape[0], self.image_shape[1]


class CNNBase(nn.Module):
    def __init__(self, image_shape, action_space, hidden_size=256):
        super().__init__()

        self.image_shape = image_shape

        self.main = nn.Sequential(
            nn.Conv2d(image_shape[2], 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1), nn.ReLU(), )
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 7 * 7, hidden_size), nn.ReLU())

        for l in self.main:
            if type(l) == nn.Conv2d:
                nn.init.orthogonal_(l.weight, nn.init.calculate_gain('relu'))
                l.bias.data.zero_()
        for l in self.fc_layers:
            if type(l) == nn.Linear:
                nn.init.orthogonal_(l.weight)
                l.bias.data.zero_()

        self.critic_linear = nn.Linear(hidden_size, 1)
        nn.init.orthogonal_(self.critic_linear.weight)
        nn.init.constant_(self.critic_linear.bias, 0.0)

        self.policy_logits = nn.Linear(hidden_size, action_space.n)
        nn.init.orthogonal_(self.policy_logits.weight, 0.01)
        nn.init.constant_(self.policy_logits.bias, 0.0)

    def forward(self, inputs):
        x = self.main(inputs.float() / 255.)
        x = x.view(-1, 32 * 7 * 7)
        x = self.fc_layers(x)
        actor = torch.distributions.Categorical(logits=self.policy_logits(x))
        critic = self.critic_linear(x)
        return actor, critic

    def image_input_shape(self):
        return self.image_shape[2], self.image_shape[0], self.image_shape[1]


class ActorCritic(nn.Module):
    def __init__(self, image_shape, action_space):
        super(ActorCritic, self).__init__()
        self.image_shape = image_shape
        observation_shape = self.image_input_shape()
        n_actions = action_space.n

        self.features = nn.Sequential(
            nn.Conv2d(observation_shape[0], 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1),
            nn.ReLU()
        )

        # self.gru = nn.GRUCell(32 * 7 * 7, 256)
        self.linear = nn.Linear(32 * 7 * 7, 256)
        self.actor = nn.Linear(256, n_actions)
        self.critic = nn.Linear(256, 1)

        for m in self.features:
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, nn.init.calculate_gain('relu'))
                nn.init.constant_(m.bias, 0.0)

        nn.init.orthogonal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0.0)

        nn.init.orthogonal_(self.critic.weight)
        nn.init.constant_(self.critic.bias, 0.0)

        nn.init.orthogonal_(self.actor.weight, 0.01)
        nn.init.constant_(self.actor.bias, 0.0)

    def forward(self, x):
        x = x / 255.
        x = self.features(x)
        x = x.view(-1, 32 * 7 * 7)
        x = self.linear(x)
        x = F.relu(x)
        # hx = self.gru(x, hx)
        actor = torch.distributions.Categorical(logits=self.actor(x))
        critic = self.critic(x)
        return actor, critic, torch.Tensor([0.])

    def image_input_shape(self):
        return self.image_shape[2], self.image_shape[0], self.image_shape[1]


def image_reshaping_layer(image, shape, greyscale=True):
    pass


class ObjectDetectionNetwork(nn.Module):
    def __init__(self, image_shape, action_space, monet_config, herke=False, load_pretrained=False):
        super().__init__()
        self.image_shape = image_shape
        self.herke = herke
        observation_shape = self.image_input_shape()
        n_actions = action_space.n
        
        # monet path
        self.object_detection = load_monet(monet_config, load_pretrained=load_pretrained)
        graph_shape = self.object_detection.module.graph_depth
        self.graph_shape = graph_shape
        self.num_objects = self.object_detection.module.num_slots

        self.interactions_linear = nn.Sequential(
                nn.Linear(graph_shape, graph_shape),
                nn.ReLU())
        self.interactions_attention = nn.Sequential(
                nn.Linear(graph_shape, graph_shape),
                nn.Sigmoid())
        self.embeddings_linear = nn.Sequential(
                nn.Linear(graph_shape, graph_shape),
                nn.ReLU())
        self.joint_graph_embedding = nn.Sequential(
                nn.Linear(graph_shape, 64),
                nn.ReLU())
        self.time_combination = nn.Sequential(
                nn.Linear(4 * 64, 256),
                nn.ReLU())
        # torch.nn.init.normal_(self.time_combination[0].bias, mean=0, std=0.0001)
        # torch.nn.init.normal_(self.time_combination[0].weight, mean=0, std=0.0001)
        
        # image conv path
        self.image_features = nn.Sequential(
            nn.Conv2d(observation_shape[0], 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 12 * 12, 256),
            nn.ReLU())

        # joint path
        self.cat_reduction = nn.Sequential(
            nn.Linear(2 * 256, 256),
            nn.ReLU())
        self.attention_layer = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * 256), 
            nn.Sigmoid())
        self.joint_linear = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU())
        self.actor = nn.Linear(256, n_actions)
        self.critic = nn.Linear(256, 1)
        
        self._loss = torch.Tensor([0.]).cuda()

    def forward(self, 
            x, 
            use_image_model=True,
            gradient_through_image=True,
            only_image_model=True,
            replace_relational_with_noise=False,
            replace_image_with_noise=False):
        if only_image_model and not use_image_model:
            raise ValueError('Cannot only use image model and also not use it')
        # normal image conv net path
        batch_size = x.shape[0]
        x = x / 255.
        
        _loss = self._loss

        # object detection path
        if not (only_image_model or replace_relational_with_noise):
            if gradient_through_image:
                interactions, embeddings, _loss = self.object_detection.module.build_image_graph(
                    x.view(-1, 3, x.shape[2], x.shape[3]))
            else:
                interactions, embeddings, _loss = self.object_detection.module.build_image_graph(
                    x.view(-1, 3, x.shape[2], x.shape[3]))
                interactions = interactions.detach()
                embeddings = embeddings.detach()
            embeddings = self.embeddings_linear(embeddings)
            interactions = self.interactions_linear(interactions)
            attention = self.interactions_attention(interactions)
            interactions = torch.sum(interactions * attention, 2)
            full_embedding = embeddings + interactions
            full_embedding = self.joint_graph_embedding(full_embedding)
            graph_embeddings = torch.sum(full_embedding, 1)
            graph_embeddings = graph_embeddings.view(batch_size, -1)
            graph_embeddings = self.time_combination(graph_embeddings)
            
        # standard model
        if use_image_model:
            img_features = self.image_features(x)
            if replace_image_with_noise:
                means = torch.zeros_like(img_features)
                img_features = torch.distributions.Normal(means, 0.01).sample()

            # path for simple image model
            if only_image_model:
                if self.herke:
                    means = torch.zeros_like(img_features)
                    noisy_img_features = img_fatures + torch.distributions.Normal(means, 1.).sample()
                    joint = torch.cat([img_featuers, noisy_img_features], -1)
                    attention = self.attention_layer(img_features)
                    joint = self.cat_reduction(joint * attention)
                joint = img_features
            
            # path for joint embedding
            else:
                if replace_relational_with_noise:
                    means = torch.zeros_like(img_features)
                    graph_embeddings = torch.distributions.Normal(means, 0.01).sample()
                joint = img_features + graph_embeddings
        else:
            joint = graph_embeddings
        joint_embedding = self.joint_linear(joint)

        # calculate actor and critic
        actor = self.actor(joint_embedding)
        actor = torch.distributions.Categorical(logits=actor)
        critic = self.critic(joint_embedding)
        return actor, critic, _loss

    def image_input_shape(self):
        return self.image_shape[2], self.image_shape[0], self.image_shape[1]

    def reset_loss(self):
        self.loss = torch.Tensor([0.]).cuda()


def load_monet(monet_config, device_id=0, load_pretrained=False):
    monet = spatial_monet.MaskedAIR(**monet_config._asdict())
    torch.cuda.set_device(device_id)
    sum([param.nelement() for param in monet.parameters()])
    monet = monet.cuda()
    monet = nn.DataParallel(monet, device_ids=[device_id])
    if load_pretrained:
        monet.load_state_dict(torch.load('pretrained_monet'))
        monet = monet.cuda()
    return monet


def down_sample_greyscale(x):
    return x
