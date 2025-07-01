import os, sys, json, pdb
import logging
import argparse
import random
import numpy as np
# from mpi4py import MPI
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import numpy as np

import random
import torch
torch.cuda.init()
from mpi4py import MPI
# FIX random seed
random_state = 0
torch.manual_seed(random_state)
from torch_geometric.data import Data
from torch_geometric.transforms import RadiusGraph, Distance, AddLaplacianEigenvectorPE

import hydragnn
from hydragnn.utils.profiling_and_tracing.time_utils import Timer
from hydragnn.utils.model import print_model
from hydragnn.utils.descriptors_and_embeddings.atomicdescriptors import (
    atomicdescriptors
)
from hydragnn.utils.descriptors_and_embeddings.chemicaldescriptors import ChemicalFeatureEncoder
from hydragnn.utils.descriptors_and_embeddings.topologicaldescriptors import compute_topo_features
from hydragnn.utils.descriptors_and_embeddings.topologicaldescriptors import *
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

def reverse_dict(input_dict):
    """Reverses a dictionary, swapping keys and values."""
    return {value: key for key, value in input_dict.items()}

# FIXME: this radis cutoff overwrites the radius cutoff currently written in the JSON file
create_graph_fromXYZ = RadiusGraph(r=5.0)  # radius cutoff in angstrom
compute_edge_lengths = Distance(norm=False, cat=True)

periodic_table = generate_dictionary_elements()
reverse_pt = reverse_dict(periodic_table)

def get_atomic_number(symbol):
    return reverse_pt.get(symbol)

# # Update each sample prior to loading.
# def niaid_pre_transform(data, transform):
#     # LPE
#     data = transform(data)

#     # gps requires relative edge features, introduced rel_lapPe as edge encodings
#     source_pe = data.pe[data.edge_index[0]]
#     target_pe = data.pe[data.edge_index[1]]
#     data.rel_pe = torch.abs(source_pe - target_pe)  # Compute feature-wise difference
#     return data

class niaid(AbstractBaseDataset):
    def __init__(
        self, datadir, num_laplacian_eigs
    ):
        super().__init__()

        df = pd.read_parquet(datadir+'niaid.parquet.gzip')

        # Chemical encoder
        ChemEncoder = ChemicalFeatureEncoder()

        # LPE
        transform = AddLaplacianEigenvectorPE(
                k=num_laplacian_eigs,
                attr_name="pe",
                is_undirected=True,
            )

        pbar = tqdm(total=df.shape[0],desc="Pre-processing data and adding chemical encodings")
        for _, row in df.iterrows():
            # Get coordinates
            a, b, c, alpha, beta, gamma = float(row['_cell_length_a'].squeeze()), float(row['_cell_length_b'].squeeze()), float(row['_cell_length_c'].squeeze()), float(row['_cell_angle_alpha'].squeeze()), float(row['_cell_angle_beta'].squeeze()), float(row['_cell_angle_gamma'].squeeze())
            pos_x, pos_y, pos_z = row[['_atom_site_fract_x']].squeeze().astype(float), row[['_atom_site_fract_y']].squeeze().astype(float), row[['_atom_site_fract_z']].squeeze().astype(float)
            pos = np.concatenate([np.expand_dims(pos_x,axis=0),np.expand_dims(pos_y,axis=0),np.expand_dims(pos_z,axis=0)],axis=0)

            # Apply transformation
            M = self.transformation_matrix(a, b, c, alpha, beta, gamma)
            pos = M @ pos
            pos = pos.T

            # node attributes
            atomic_numbers = np.array([get_atomic_number(atom) for atom in row['_atom_site_type_symbol']]).astype(float)
            # if row['Filename'] == 'DB0-m3_o7_o25_f0_nbo.sym.30_repeat':
            #     pdb.set_trace()
            partial_charges = np.array(row['_atom_type_partial_charge']).astype(float)
            x = np.concatenate([np.expand_dims(atomic_numbers,axis=1),np.expand_dims(partial_charges,axis=1)],axis=1)

            # Create the PyTorch Geometric Data object
            data = Data()

            data.x = torch.Tensor(x)
            data.pos = torch.Tensor(pos)

            # Create graph
            data = create_graph_fromXYZ(data)
            # Add edge length as edge feature
            data = compute_edge_lengths(data)
            data.edge_attr = data.edge_attr.to(torch.float32)

            # Pre-transform
            try:
                data = transform(data) #lapPE
                # data = niaid_pre_transform(data, transform)
                data = ChemEncoder.compute_chem_features(data)
                data = compute_topo_features(data)

                has_nan = torch.isnan(data.pe).any() + torch.isnan(data.rel_pe).any()
                if has_nan:
                    raise ValueError("NaN persists")

                self.dataset.append(data)
            except:
                print(row['Filename'])
            pbar.update(1)
        pbar.close()

        # self.get_topo_encodings()

        random.shuffle(self.dataset)

    def get_topo_encodings(self):
        n_procs = min(cpu_count(), len(self.dataset))
        chunksize = max(1, len(self.dataset) // (n_procs * 4))  # tune this

        with Pool(processes=n_procs) as pool:
            iterator = pool.imap(compute_topo_features, self.dataset, chunksize)
            self.dataset = list(tqdm(iterator, total=len(self.dataset), desc="Adding topological encodings"))
            # self.dataset = pool.map(compute_topo_features, self.dataset, chunksize)

    def transformation_matrix(self, a, b, c, alpha, beta, gamma):
        # Convert angles to radians
        alpha1 = np.radians(alpha)
        beta = np.radians(beta)
        gamma = np.radians(gamma)

        # Trigonometric functions
        sin1, sin2 = np.sin(alpha1), np.sin(beta)
        cos1, cos2, cos3 = np.cos(alpha1), np.cos(beta), np.cos(gamma)
        cot1, cot2 = 1/np.tan(alpha1), 1/np.tan(beta)
        csc1, csc2 = 1/np.sin(alpha1), 1/np.sin(beta)

        # First row, first column element
        term1 = cot1 * cot2 - csc1 * csc2 * cos3
        R11 = a * sin2 * np.sqrt(1 - term1**2)

        # Transformation matrix
        M = np.array([
            [R11, 0, 0],
            [a * csc1 * cos3 - a * cot1 * cos2, b * sin1, 0],
            [a * cos2, b * cos1, c]
        ])

        return M

    def len(self):
        return len(self.dataset)

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
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "niaid.json")
    with open(filename, "r") as f:
        config = json.load(f)

    ##################################################################################################################
    # Always initialize for multi-rank training.
    comm_size, rank = hydragnn.utils.distributed.setup_ddp()
    ##################################################################################################################

    comm = MPI.COMM_WORLD

    modelname = "niaid" 

    if preonly:
        ## local data
        dataset = niaid(
            'dataset/raw/', config["NeuralNetwork"]["Architecture"]["num_laplacian_eigs"],
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

    log_name = f"niaid_test_{mpnn_type}" if mpnn_type else "niaid_test"
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
    config["NeuralNetwork"]["Architecture"]["ce_dim"] = trainset[0].ce.shape[1]
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
        description="Run the niaid example with optional model type."
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
