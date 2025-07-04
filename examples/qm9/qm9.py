import os, sys, json, pdb, math
import logging
import argparse
import random
from mpi4py import MPI
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import torch
# torch.cuda.init()
# from mpi4py import MPI
# FIX random seed
random_state = 0
torch.manual_seed(random_state)
from torch_geometric.transforms import AddLaplacianEigenvectorPE
from torch_geometric.datasets import QM9
from torch_geometric.transforms import AddLaplacianEigenvectorPE

import hydragnn
from hydragnn.utils.profiling_and_tracing.time_utils import Timer
from hydragnn.utils.model import print_model
# from hydragnn.utils.descriptors_and_embeddings.atomicdescriptors import (
#     atomicdescriptors,
# )
from hydragnn.utils.descriptors_and_embeddings.chemicaldescriptors import ChemicalFeatureEncoder
from hydragnn.utils.descriptors_and_embeddings.topologicaldescriptors import compute_topo_features
from hydragnn.utils.datasets.abstractbasedataset import AbstractBaseDataset
from hydragnn.utils.datasets.distdataset import DistDataset
from hydragnn.utils.datasets.pickledataset import (
    SimplePickleWriter,
    SimplePickleDataset,
)
from hydragnn.preprocess.graph_samples_checks_and_updates import gather_deg
from hydragnn.preprocess.load_data import split_dataset
import hydragnn.utils.profiling_and_tracing.tracer as tr

try:
    from hydragnn.utils.datasets.adiosdataset import AdiosWriter, AdiosDataset
except ImportError:
    pass

from generate_dictionaries_pure_elements import (
    generate_dictionary_elements,
)

def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))

class QM9_enc(AbstractBaseDataset):
    def __init__(
        self, datadir, num_laplacian_eigs
    ):
        super().__init__()

        # Chemical encoder
        ChemEncoder = ChemicalFeatureEncoder()

        self.trainset, self.valset, self.testset = [], [], []

        # LPE
        transform = AddLaplacianEigenvectorPE(
            k=num_laplacian_eigs,
            attr_name="pe",
            is_undirected=True,
        )

        qm9_dataset = QM9(root=datadir)

        for data in tqdm(qm9_dataset, total=len(qm9_dataset), desc="Preprocessing and Encoding"):
            # Encoders
            data.x = data.z.float().view(-1, 1)
            data.y = data.y[:, 10] / len(data.x)
            del data.z
            del data.smiles
            del data.name
            del data.idx
            data = ChemEncoder.compute_chem_features(data)
            data = transform(data) #lapPE
            data = compute_topo_features(data)
            self.dataset.append( data )

    def len(self):
        return( len(self.dataset) )

    def get(self, idx):
        return self.dataset[idx]
    
def main(preonly=False, format='pickle', ddstore=False, 
        ddstore_width=None, shmem=False, 
        mpnn_type=None, global_attn_engine=None, 
        global_attn_type=None):

    # FIX random seed
    random_state = 0
    torch.manual_seed(random_state)

    # Set this path for output.
    try:
        os.environ["SERIALIZED_DATA_PATH"]
    except KeyError:
        os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()

    # Configurable run choices (JSON file that accompanies this example script).
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qm9.json")
    with open(filename, "r") as f:
        config = json.load(f)

    ##################################################################################################################
    # Always initialize for multi-rank training.
    comm_size, rank = hydragnn.utils.distributed.setup_ddp()
    ##################################################################################################################

    comm = MPI.COMM_WORLD

    modelname = "qm9" 

    if preonly:
        ## local data
        dataset = QM9_enc(
            'dataset/raw', config["NeuralNetwork"]["Architecture"]["num_laplacian_eigs"],
        )

        ## This is a local split
        trainset, valset, testset = split_dataset(
            dataset=dataset,
            perc_train=0.8,
            stratify_splitting=False,
        )

        print("Local splitting: ", len(trainset), len(valset), len(testset))

        deg = gather_deg(trainset)
        config["pna_deg"] = deg

        ## adios
        if format == "adios":
            fname = os.path.join(
                os.path.dirname(__file__), "./dataset/%s.bp" % modelname
            )
            adwriter = AdiosWriter(fname, comm)
            adwriter.add("trainset", trainset)
            adwriter.add("valset", valset)
            adwriter.add("testset", testset)
            adwriter.add_global("pna_deg", deg)
            adwriter.save()

        ## pickle
        elif format == "pickle":
            basedir = os.path.join(
                os.path.dirname(__file__), "dataset", "%s.pickle" % modelname
            )

            attrs = dict()
            attrs["pna_deg"] = deg

            SimplePickleWriter(
                trainset,
                basedir,
                "trainset",
                use_subdir=True,
                attrs=attrs,
            )
            SimplePickleWriter(
                valset,
                basedir,
                "valset",
                use_subdir=True,
            )
            SimplePickleWriter(
                testset,
                basedir,
                "testset",
                use_subdir=True,
            )
        sys.exit(0)

    # If a model type is provided, update the configuration accordingly.
    if global_attn_engine:
        config["NeuralNetwork"]["Architecture"][
            "global_attn_engine"
        ] = global_attn_engine

    if global_attn_type:
        config["NeuralNetwork"]["Architecture"]["global_attn_type"] = global_attn_type

    if mpnn_type:
        config["NeuralNetwork"]["Architecture"]["mpnn_type"] = mpnn_type

    verbosity = config["Verbosity"]["level"]
    var_config = config["NeuralNetwork"]["Variables_of_interest"]

    # Always initialize for multi-rank training.
    world_size, world_rank = hydragnn.utils.distributed.setup_ddp()

    log_name = f"qm9_test_{mpnn_type}" if mpnn_type else "qm9_test"
    # Enable print to log file.
    hydragnn.utils.print.print_utils.setup_log(log_name)

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
    config["NeuralNetwork"]["Architecture"]["pe_dim"] = trainset[0].pe.shape[1]
    config["NeuralNetwork"]["Architecture"]["ce_dim"] = 0
    config["NeuralNetwork"]["Architecture"]["rel_pe_dim"] = trainset[0].rel_pe.shape[1]

    if ddstore:
        os.environ["HYDRAGNN_AGGR_BACKEND"] = "mpi"
        os.environ["HYDRAGNN_USE_ddstore"] = "1"

    (train_loader, val_loader, test_loader,) = hydragnn.preprocess.create_dataloaders(
        trainset, valset, testset, config["NeuralNetwork"]["Training"]["batch_size"]
    )

    ## Good to sync with everyone right after DDStore setup
    comm.Barrier()

    if args.ddstore:
        train_loader.dataset.ddstore.epoch_begin()
    config = hydragnn.utils.input_config_parsing.update_config(
        config, train_loader, val_loader, test_loader
    )
    if args.ddstore:
        train_loader.dataset.ddstore.epoch_end()
    ## Good to sync with everyone right after DDStore setup
    comm.Barrier()
    
    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"],
        verbosity=verbosity,
    )
    model = hydragnn.utils.distributed.get_distributed_model(model, verbosity)

    # Print details of neural network architecture
    print_model(model)

    learning_rate = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
    )

    # Run training with the given model and qm9 datasets.
    writer = hydragnn.utils.model.model.get_summary_writer(log_name)
    hydragnn.utils.input_config_parsing.save_config(config, log_name)

    hydragnn.train.train_validate_test(
        model,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
        writer,
        scheduler,
        config["NeuralNetwork"],
        log_name,
        verbosity,
        create_plots=False
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the qm9 example with optional model type."
    )
    parser.add_argument(
        "--preonly",
        action="store_true",
        help="preprocess only (no training)",
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
    parser.add_argument(
        "--mpnn_type",
        type=str,
        default=None,
        help="Specify the model type for training (default: None).",
    )
    parser.add_argument(
        "--global_attn_engine",
        type=str,
        default=None,
        help="Specify if global attention is being used (default: None).",
    )
    parser.add_argument(
        "--global_attn_type",
        type=str,
        default=None,
        help="Specify the global attention type (default: None).",
    )
    args = parser.parse_args()
    main(preonly=args.preonly, format=args.format, ddstore=args.ddstore, 
        ddstore_width=args.ddstore_width, shmem=args.shmem, mpnn_type=args.mpnn_type, 
        global_attn_engine=args.global_attn_engine, global_attn_type=args.global_attn_type)














# import os
# import pdb
# import json
# import torch
# import torch_geometric
# from torch_geometric.transforms import AddLaplacianEigenvectorPE
# import argparse

# # deprecated in torch_geometric 2.0
# try:
#     from torch_geometric.loader import DataLoader
# except ImportError:
#     from torch_geometric.data import DataLoader

# import hydragnn

# num_samples = int(1e7) #TODO:change to 10000 before merge

# # Update each sample prior to loading.
# def qm9_pre_transform(data, transform):
#     # LPE
#     data = transform(data)
#     # Set descriptor as element type.
#     data.x = data.z.float().view(-1, 1)
#     # Only predict free energy (index 10 of 19 properties) for this run.
#     data.y = data.y[:, 10] / len(data.x)
#     graph_features_dim = [1]
#     node_feature_dim = [1]
#     # gps requires relative edge features, introduced rel_lapPe as edge encodings
#     source_pe = data.pe[data.edge_index[0]]
#     target_pe = data.pe[data.edge_index[1]]
#     data.rel_pe = torch.abs(source_pe - target_pe)  # Compute feature-wise difference
#     return data


# def qm9_pre_filter(data):
#     return data.idx < num_samples


# def main(mpnn_type=None, global_attn_engine=None, global_attn_type=None):
#     # FIX random seed
#     random_state = 0
#     torch.manual_seed(random_state)

#     # Set this path for output.
#     try:
#         os.environ["SERIALIZED_DATA_PATH"]
#     except KeyError:
#         os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()

#     # Configurable run choices (JSON file that accompanies this example script).
#     filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qm9.json")
#     with open(filename, "r") as f:
#         config = json.load(f)

#     # If a model type is provided, update the configuration accordingly.
#     if global_attn_engine:
#         config["NeuralNetwork"]["Architecture"][
#             "global_attn_engine"
#         ] = global_attn_engine

#     if global_attn_type:
#         config["NeuralNetwork"]["Architecture"]["global_attn_type"] = global_attn_type

#     if mpnn_type:
#         config["NeuralNetwork"]["Architecture"]["mpnn_type"] = mpnn_type

#     verbosity = config["Verbosity"]["level"]
#     var_config = config["NeuralNetwork"]["Variables_of_interest"]

#     # Always initialize for multi-rank training.
#     world_size, world_rank = hydragnn.utils.distributed.setup_ddp()

#     log_name = f"qm9_test_{mpnn_type}" if mpnn_type else "qm9_test"
#     # Enable print to log file.
#     hydragnn.utils.print.print_utils.setup_log(log_name)

#     # LPE
#     transform = AddLaplacianEigenvectorPE(
#         k=config["NeuralNetwork"]["Architecture"]["pe_dim"],
#         attr_name="pe",
#         is_undirected=True,
#     )

#     # Use built-in torch_geometric datasets.
#     # Filter function above used to run quick example.
#     # NOTE: data is moved to the device in the pre-transform.
#     # NOTE: transforms/filters will NOT be re-run unless the qm9/processed/ directory is removed.
#     dataset = torch_geometric.datasets.QM9(
#         root="dataset/qm9",
#         pre_transform=lambda data: qm9_pre_transform(data, transform),
#         pre_filter=qm9_pre_filter,
#     )
#     train, val, test = hydragnn.preprocess.split_dataset(
#         dataset, config["NeuralNetwork"]["Training"]["perc_train"], False
#     )

#     (train_loader, val_loader, test_loader,) = hydragnn.preprocess.create_dataloaders(
#         train, val, test, config["NeuralNetwork"]["Training"]["batch_size"]
#     )

#     config = hydragnn.utils.input_config_parsing.update_config(
#         config, train_loader, val_loader, test_loader
#     )

#     model = hydragnn.models.create_model_config(
#         config=config["NeuralNetwork"],
#         verbosity=verbosity,
#     )
#     model = hydragnn.utils.distributed.get_distributed_model(model, verbosity)

#     learning_rate = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
#     optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
#     )

#     # Run training with the given model and qm9 datasets.
#     writer = hydragnn.utils.model.model.get_summary_writer(log_name)
#     hydragnn.utils.input_config_parsing.save_config(config, log_name)

#     hydragnn.train.train_validate_test(
#         model,
#         optimizer,
#         train_loader,
#         val_loader,
#         test_loader,
#         writer,
#         scheduler,
#         config["NeuralNetwork"],
#         log_name,
#         verbosity,
#         create_plots=True
#     )


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Run the QM9 example with optional model type."
#     )
#     parser.add_argument(
#         "--mpnn_type",
#         type=str,
#         default=None,
#         help="Specify the model type for training (default: None).",
#     )
#     parser.add_argument(
#         "--global_attn_engine",
#         type=str,
#         default=None,
#         help="Specify if global attention is being used (default: None).",
#     )
#     parser.add_argument(
#         "--global_attn_type",
#         type=str,
#         default=None,
#         help="Specify the global attention type (default: None).",
#     )
#     args = parser.parse_args()
#     main(mpnn_type=args.mpnn_type)
