import os, json, pdb
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
# from mpi4py import MPI
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score

import torch
torch.cuda.init()
from mpi4py import MPI
# FIX random seed
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

def main(dir_path, format='pickle', ddstore=False, 
        ddstore_width=None, shmem=False):
    
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
    filename = os.path.join(dir_path+"/config.json")
    with open(filename, "r") as f:
        config = json.load(f)

    ##################################################################################################################
    # Always initialize for multi-rank training.
    comm_size, rank = hydragnn.utils.distributed.setup_ddp()
    ##################################################################################################################

    comm = MPI.COMM_WORLD

    modelname = "pcba" 

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

    # Update encoding dimensions
    config["NeuralNetwork"]["Architecture"]["lpe_dim"] = trainset[0].lpe.shape[1]
    config["NeuralNetwork"]["Architecture"]["pe_dim"] = trainset[0].pe.shape[1]
    config["NeuralNetwork"]["Architecture"]["ce_dim"] = trainset[0].ce.shape[1]
    config["NeuralNetwork"]["Architecture"]["rel_pe_dim"] = trainset[0].rel_pe.shape[1]

    if ddstore:
        os.environ["HYDRAGNN_AGGR_BACKEND"] = "mpi"
        os.environ["HYDRAGNN_USE_ddstore"] = "1"

    # Batch size for current system
    with open("pcba.json", "r") as f:
        config_sys = json.load(f)

    (train_loader, val_loader, test_loader,) = hydragnn.preprocess.create_dataloaders(
        trainset, valset, testset, config_sys["NeuralNetwork"]["Training"]["batch_size"]
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
    
    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"],
        verbosity=verbosity,
    )
    model = hydragnn.utils.distributed.get_distributed_model(model, verbosity)

    # # Print details of neural network architecture
    # print_model(model)

    # Load model weights from checkpoint
    model = load_existing_model(model, path=dir_path)

    # Inference mode
    model.eval()

    all_scores = []
    all_labels = []
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Multi-label Inference"):
            data = data.to(device)
            scores = model(data)
            if isinstance(scores, (list, tuple)):
                scores = scores[0]
            all_scores.append(scores.cpu())
            y = torch.reshape(data.y,[-1,128])
            all_labels.append(y.cpu())

    scores = torch.cat(all_scores, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()

    # per-label AP
    n_labels = labels.shape[1]
    ap_per_label = []
    for i in range(n_labels):
        y_true = labels[:, i]
        y_score = scores[:, i]

        # mask out any samples where the ground-truth is NaN
        mask = ~np.isnan(y_true)
        if mask.sum() == 0:
            # no valid samples for this label → set to NaN or 0.0 as you prefer
            ap = float('nan')
        else:
            ap = average_precision_score(y_true[mask], y_score[mask])
        ap_per_label.append(ap)

    # macro-average over all non-NaN APs
    valid_aps = [a for a in ap_per_label if not np.isnan(a)]
    ap_macro = np.mean(valid_aps) if valid_aps else float('nan')

    if rank == 0:
        for i, ap in enumerate(ap_per_label):
            print(f"Label {i:3d} AP: {ap:.4f}")
        print(f"\nAverage Precision (macro): {ap_macro:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run HydraGNN Inference."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--adios",
        help="Adios dataset",
        action="store_const",
        dest="format",
        const="adios",
    )
    group.add_argument(
        "--pickle",
        help="Pickle dataset",
        action="store_const",
        dest="format",
        const="pickle",
    )
    parser.set_defaults(format="pickle")
    parser.add_argument(
        "--ddstore",
        action="store_true", 
        help="ddstore dataset"
    )
    parser.add_argument(
        "--ddstore_width", 
        type=int, 
        help="ddstore width", 
        default=None
    )
    parser.add_argument(
        "--shmem", 
        action="store_true", 
        help="shmem"
    )
    args = parser.parse_args()

    # dir_path = 'hpo_backup/exp4_new/logs/pcba_hpo_trials_0.141'
    dir_path = 'logs/pcba_exp4'

    main(dir_path, format=args.format, ddstore=args.ddstore, 
        ddstore_width=args.ddstore_width, shmem=args.shmem)
