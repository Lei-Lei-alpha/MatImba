import sys
from .losses import *
from torch.optim import *
from torch.optim.lr_scheduler import *

import numpy as np
import pandas as pd

def get_obj(obj_name):
    """
    Return object from the object name
    """
    from ..models import gbr_ensemble
    from ..models.resnet import ResNet
    from ..models.megnet import MEGNet
    return getattr(sys.modules[__name__], obj_name)

def load_model(ckpt, **params):
    saved_states = torch.load(ckpt)
    model = get_obj(saved_states["model"]["name"])(**params)
    model.load_state_dict(saved_states["model"]["states"])
    return model