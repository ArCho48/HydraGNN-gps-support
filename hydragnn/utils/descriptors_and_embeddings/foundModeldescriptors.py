import pdb, os
import numpy as np
from collections import OrderedDict
import torch
# FIX random seed
random_state = 0
torch.manual_seed(random_state)
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

import hydragnn
from hydragnn.utils.model import print_model
from hydragnn.utils.datasets.distdataset import DistDataset
from hydragnn.utils.datasets.pickledataset import (
    SimplePickleDataset,
)
from hydragnn.utils.print.print_utils import print_master
import hydragnn.utils.profiling_and_tracing.tracer as tr
from hydragnn.utils.distributed import (
    get_device_name,
)
try:
    from hydragnn.utils.datasets.adiosdataset import AdiosDataset
except ImportError:
    pass

def load_model(model, path):
    path_name = os.path.join(path, path.split('/')[-1] + ".pk")
    map_location = {"cuda:%d" % 0: get_device_name()}
    print_master("Load existing model:", path_name)
    checkpoint = torch.load(path_name, map_location=map_location)
    state_dict = checkpoint["model_state_dict"]

    if not next(iter(state_dict)).startswith("module"):
        ddp_state_dict = OrderedDict()
        for k, v in state_dict.items():
            k = "module." + k
            ddp_state_dict[k] = v
        state_dict = ddp_state_dict
    model.load_state_dict(state_dict)

    return model