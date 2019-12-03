from spatial_monet.spatial_monet import MaskedAIR

def get_air(model_config):
    return MaskedAIR(model_config._asdict())
