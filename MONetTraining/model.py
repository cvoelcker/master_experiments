from spatial_monet.spatial_monet import MaskedAIR
from spatial_monet.monet import Monet


def get_air(model_config, baseline=None):
    if baseline is None:
        return MaskedAIR(model_config._asdict())
    elif baseline == 'monet':
        return Monet(model_config._asdict())
