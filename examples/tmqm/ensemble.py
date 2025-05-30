import os, json
import sys, pdb
import pickle
import random
import numpy as np
import pandas as pd
from mpi4py import MPI
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from tqdm import tqdm
import torch
# torch.cuda.init()
# from mpi4py import MPI
# FIX random seed
import torch.optim as optim
from torch.utils.data.dataset import random_split
from torch_geometric.loader import DataLoader

import logging
import argparse
from collections import OrderedDict
from tqdm import tqdm
from scipy.stats import pearsonr

random_state = 0
torch.manual_seed(random_state)

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

from deephyper.analysis.hpo import parameters_from_row
from deephyper.ensemble.aggregator import MeanAggregator
from deephyper.ensemble.loss import SquaredError
from deephyper.ensemble.selector import GreedySelector, TopKSelector
from deephyper.predictor.torch import TorchPredictor
from deephyper.ensemble import EnsemblePredictor

def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))

def load_existing_model(model, path):
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GraphTorchPredictor(TorchPredictor):
    """Represents a frozen torch model that can only predict."""

    def __init__(self, module: torch.nn.Module):
        if not isinstance(module, torch.nn.Module):
            raise ValueError(
                f"The given module is of type {type(module)} when it should be of type "
                f"torch.nn.Module!"
            )
        self.module = module

    def pre_process_inputs(self, batch):
        return batch

    def post_process_predictions(self, y):
        y = y[0].detach().cpu().numpy()
        return y

    def predict(self, X):
        X = self.pre_process_inputs(X)
        training = self.module.training
        if training:
            self.module.eval()

        if hasattr(self.module, "predict_proba"):
            y = self.module.predict_proba(X)
        else:
            y = self.module(X)

        self.module.train(training)
        y = self.post_process_predictions(y)
        return y


def create_ensemble(
    model_checkpoints_dir,
    verbosity,
    val_loader,
    ensemble_selector: str = "topk",
    k=10,
):
    y_predictors = []
    valid_y = []
    job_id_predictors = []

    for model_name in tqdm(os.listdir(model_checkpoints_dir)):
        if 'tmqm' not in model_name:
            continue
        job_id = int(model_name.split('_')[-1].split(".")[-1])

        # Configurable run choices (JSON file that accompanies this example script).
        filename = os.path.join(model_checkpoints_dir,model_name+"/config.json")
        with open(filename, "r") as f:
            config = json.load(f)

        model = hydragnn.models.create_model_config(
            config=config["NeuralNetwork"],
            verbosity=verbosity,
        )
        model = hydragnn.utils.distributed.get_distributed_model(model, verbosity)

        # Load model weights from checkpoint
        model = load_existing_model(model, path=os.path.join(model_checkpoints_dir,model_name))

        # Inference mode
        model.eval()

        with torch.no_grad():
            batch_y_pred = []
            batch_y_valid = []

            for batch in val_loader:
                batch = batch.to(device)
                batch_y_pred.append(model(batch)[0].detach().cpu().numpy())
                batch_y_valid.append(batch.y.cpu().numpy())

        y_predictors.append(np.concatenate(batch_y_pred))
        valid_y.append(np.concatenate(batch_y_valid))
        job_id_predictors.append(job_id)

    y_predictors = np.array(y_predictors)
    valid_y = np.array(valid_y)

    ## Use TopK or Greedy/Caruana
    if ensemble_selector == "topk":
        selector = TopKSelector(
            loss_func=SquaredError(),
            k=k,
        )
    else:
        selector = GreedySelector(
            loss_func=SquaredError(),
            aggregator=MeanAggregator(),
            k=k,
            max_it=200,
            k_init=10,
            early_stopping=False,
            with_replacement=True,
            bagging=True,
            verbose=False,
        )

    selected_predictors_indexes, selected_predictors_weights = selector.select(
        valid_y,
        y_predictors,
    )

    selected_predictors_job_ids = np.array(job_id_predictors)[
        selected_predictors_indexes
    ]

    return (selected_predictors_job_ids, selected_predictors_weights)


def main(dir_path, format='pickle', ddstore=False, 
        ddstore_width=None, shmem=False):
    
    # Ensmeble type
    ensem_type = sys.argv[1]

    # Path 
    dir_path = dir_path+'ensmeble_'+ensem_type

    # Create output directory if it doesn't exist
    os.makedirs(dir_path, exist_ok=True)

    # Determine device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # FIX random seed
    random_state = 0
    torch.manual_seed(random_state)

    # Set this path for output.
    try:
        os.environ["SERIALIZED_DATA_PATH"]
    except KeyError:
        os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()

    # Configurable run choices (JSON file that accompanies this example script).
    filename = os.path.join("./tmqm.json")
    with open(filename, "r") as f:
        config = json.load(f)

    ##################################################################################################################
    # Always initialize for multi-rank training.
    comm_size, rank = hydragnn.utils.distributed.setup_ddp()
    ##################################################################################################################

    comm = MPI.COMM_WORLD

    modelname = "tmqm" 

    verbosity = config["Verbosity"]["level"]
    var_config = config["NeuralNetwork"]["Variables_of_interest"]

    # Always initialize for multi-rank training.
    world_size, world_rank = hydragnn.utils.distributed.setup_ddp()

    if format == "adios":
        info("Adios load")
        assert not (shmem and ddstore), "Cannot use both ddstore and shmem"
        opt = {
            "preload": False,
            "shmem": shmem,
            "ddstore": ddstore,
            "ddstore_width": ddstore_width,
        }
        fname = os.path.join(os.path.dirname(__file__), "./dataset/%s.bp" % modelname)
        trainset = AdiosDataset(fname, "trainset", comm, **opt, var_config=var_config)
        valset = AdiosDataset(fname, "valset", comm, **opt, var_config=var_config)
        testset = AdiosDataset(fname, "testset", comm, **opt, var_config=var_config)
    elif format == "pickle":
        info("Pickle load")
        basedir = os.path.join(
            os.path.dirname(__file__), "dataset", "%s.pickle" % modelname
        )
        trainset = SimplePickleDataset(
            basedir=basedir, label="trainset", var_config=var_config
        )
        valset = SimplePickleDataset(
            basedir=basedir, label="valset", var_config=var_config
        )
        testset = SimplePickleDataset(
            basedir=basedir, label="testset", var_config=var_config
        )
        pna_deg = trainset.pna_deg
        if ddstore:
            opt = {"ddstore_width": ddstore_width}
            trainset = DistDataset(trainset, "trainset", comm, **opt)
            valset = DistDataset(valset, "valset", comm, **opt)
            testset = DistDataset(testset, "testset", comm, **opt)
            trainset.pna_deg = pna_deg
    else:
        raise NotImplementedError("No supported format: %s" % (format))

    info(
        "trainset,valset,testset size: %d %d %d"
        % (len(trainset), len(valset), len(testset))
    )

    if ddstore:
        os.environ["HYDRAGNN_AGGR_BACKEND"] = "mpi"
        os.environ["HYDRAGNN_USE_ddstore"] = "1"

    (train_loader, val_loader, test_loader,) = hydragnn.preprocess.create_dataloaders(
        trainset, valset, testset, config["NeuralNetwork"]["Training"]["batch_size"]
    )
    
    ## Good to sync with everyone right after DDStore setup
    comm.Barrier()

    if ddstore:
        train_loader.dataset.ddstore.epoch_begin()
    config = hydragnn.utils.input_config_parsing.update_config(
        config, train_loader, val_loader, test_loader
    )
    if ddstore:
        train_loader.dataset.ddstore.epoch_end()
    ## Good to sync with everyone right after DDStore setup
    comm.Barrier()

    # Load results and model
    model_checkpoints_dir = "HPO_wgps/logs/"

    k=10

    # selected_predictors_job_ids, selected_predictors_weights = create_ensemble( 
    #     model_checkpoints_dir, verbosity, val_loader, ensem_type, k
    # )

    selected_predictors_job_ids = [71,57,114,45,118,52,120,116,107,117]
    selected_predictors_weights = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
    # pdb.set_trace()
    predictors = []
    for index, job_id in enumerate(selected_predictors_job_ids):
        model_dir = os.path.join(model_checkpoints_dir,"tmqm_hpo_trials_0."+str(job_id))

        # Configurable run choices (JSON file that accompanies this example script).
        filename = os.path.join(model_dir+"/config.json")
        with open(filename, "r") as f:
            config = json.load(f)

        model = hydragnn.models.create_model_config(
            config=config["NeuralNetwork"],
            verbosity=verbosity,
        )
        model = hydragnn.utils.distributed.get_distributed_model(model, verbosity)

        # Load model weights from checkpoint
        model = load_existing_model(model, path=model_dir)

        # Inference mode
        model.eval()

        predictors.append(GraphTorchPredictor(model))

    ensemble = EnsemblePredictor(
        predictors=predictors,
        aggregator=MeanAggregator(with_scale=True),
        weights=selected_predictors_weights,
    )

    # Test set
    test_error = []
    y_pred_mean_store = []
    y_pred_std_store = []
    test_y = []
    for batch in test_loader:
        batch = batch.to(device)
        y_pred = ensemble.predict(batch)
        y_pred_mean_store.append(y_pred["loc"])
        y_pred_std_store.append(np.sqrt(y_pred["scale"]))
        test_y.append(batch.y.cpu().numpy())
        # pdb.set_trace()
        test_error.append(y_pred["loc"] - (batch.y.cpu().numpy()))
    test_error = np.concatenate(test_error)
    test_mse = np.mean(np.square(test_error))

    print(test_mse)

    y_pred_mean_store = np.concatenate(y_pred_mean_store)
    y_pred_std_store = np.concatenate(y_pred_std_store)
    test_y = np.concatenate(test_y)
    pdb.set_trace()
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot with error bars
    ax.errorbar(test_y, y_pred_mean_store, yerr=y_pred_std_store, fmt='o', alpha=0.7, label='Model Predictions', ecolor='gray', elinewidth=3, capsize=0)
    ax.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 'k--', lw=2, label='Perfect Prediction')
    ax.set_xlabel('$\Delta G_{ox}$ (eV) - Actual Values',fontsize=15)
    ax.set_ylabel('$\Delta G_{ox}$ (eV) - Predicted Values',fontsize=15)
    ax.set_title('GCN Ensemble ('+ sys.argv[1] +') Predictions',fontsize=17)
    ax.legend(fontsize=13)
    ax.grid(True)
    plt.tight_layout()
    name = 'fig1' if sys.argv[1] == 'greedy' else 'fig2'
    plt.savefig("../results/gcn/"+name+".png")

if __name__ == "__main__":
    dir_path = 'HPO_wgps/logs/'
    main(dir_path)
