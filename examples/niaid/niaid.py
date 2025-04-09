import os, sys, json, pdb
# os.environ["CUDA_VISIBLE_DEVICES"]="3,5"
import logging
import argparse
import random
from CifFile import ReadCif
from mpi4py import MPI
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
# FIX random seed
random_state = 0
torch.manual_seed(random_state)
from torch_geometric.data import Data
from torch_geometric.transforms import RadiusGraph, Distance, AddLaplacianEigenvectorPE


import hydragnn
from hydragnn.utils.descriptors_and_embeddings.atomicdescriptors import (
    atomicdescriptors,
)
from hydragnn.utils.datasets.abstractbasedataset import AbstractBaseDataset
from hydragnn.utils.datasets.pickledataset import (
    SimplePickleWriter,
    SimplePickleDataset,
)
from hydragnn.preprocess.graph_samples_checks_and_updates import gather_deg
from hydragnn.preprocess.load_data import split_dataset
import hydragnn.utils.profiling_and_tracing.tracer as tr

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

# atomicdescriptor = atomicdescriptors(
#     embeddingfilename="./embedding.json",
#     overwritten=True,
#     element_types=None,
#     one_hot=False,
# )

periodic_table = generate_dictionary_elements()
reverse_pt = reverse_dict(periodic_table)

def get_atomic_number(symbol):
    return reverse_pt.get(symbol)

# Update each sample prior to loading.
def niaid_pre_transform(data, transform):
    # LPE
    data = transform(data)

    # gps requires relative edge features, introduced rel_lapPe as edge encodings
    source_pe = data.pe[data.edge_index[0]]
    target_pe = data.pe[data.edge_index[1]]
    data.rel_pe = torch.abs(source_pe - target_pe)  # Compute feature-wise difference
    return data

# def add_atomic_descriptors(data):
#     descriptor_tensor = torch.empty((data.x.shape[0], 18))
#     for atom_id in range(data.x.shape[0]):
#         atomic_string = periodic_table[int(data.x[atom_id, 0].item())]
#         descriptor_tensor[atom_id, :] = atomicdescriptor.get_atom_features(
#             atomic_string
#         )
#         data.x = torch.cat([data.x, descriptor_tensor], dim=1)

#     return data

class niaid(AbstractBaseDataset):
    def __init__(
        self, datadir, pe_dim
    ):
        super().__init__()

        df = pd.read_parquet(datadir+'niaid.parquet.gzip')

         # LPE
        transform = AddLaplacianEigenvectorPE(
                k=pe_dim,
                attr_name="pe",
                is_undirected=True,
            )

        pbar = tqdm(total=df.shape[0])
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
            if row['Filename'] == 'DB0-m3_o7_o25_f0_nbo.sym.30_repeat':
                pdb.set_trace()
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
            data = niaid_pre_transform(data, transform)
            # data = add_atomic_descriptors(data)

            self.dataset.append(data)
            pbar.update(1)
        pbar.close()

        random.shuffle(self.dataset)

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

def main(preonly=False, mpnn_type=None, global_attn_engine=None, global_attn_type=None):
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

    if preonly:
        ## local data
        dataset = niaid(
            'dataset/raw/', config["NeuralNetwork"]["Architecture"]["pe_dim"],
        )
        ## This is a local split
        trainset, valset, testset = split_dataset(
            dataset=dataset,
            perc_train=0.8,
            stratify_splitting=False,
        )

        print("Local splitting: ", len(trainset), len(valset), len(testset))

        ## pickle
        basedir = os.path.join(
            os.path.dirname(__file__), "dataset", "%s.pickle" % 'niaid'
        )

        SimplePickleWriter(
            trainset,
            basedir,
            "trainset",
            # minmax_node_feature=total.minmax_node_feature,
            # minmax_graph_feature=total.minmax_graph_feature,
            use_subdir=True,
        )
        SimplePickleWriter(
            valset,
            basedir,
            "valset",
            # minmax_node_feature=total.minmax_node_feature,
            # minmax_graph_feature=total.minmax_graph_feature,
            use_subdir=True,
        )
        SimplePickleWriter(
            testset,
            basedir,
            "testset",
            # minmax_node_feature=total.minmax_node_feature,
            # minmax_graph_feature=total.minmax_graph_feature,
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

    info("Pickle load")
    basedir = os.path.join(
        os.path.dirname(__file__), "dataset", "%s.pickle" % 'niaid'
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
    
    (train_loader, val_loader, test_loader,) = hydragnn.preprocess.create_dataloaders(
        trainset, valset, testset, config["NeuralNetwork"]["Training"]["batch_size"]
    )

    config = hydragnn.utils.input_config_parsing.update_config(
        config, train_loader, val_loader, test_loader
    )

    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"],
        verbosity=verbosity,
    )
    model = hydragnn.utils.distributed.get_distributed_model(model, verbosity)

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
        create_plots=True
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the niaid example with optional model type."
    )
    parser.add_argument(
        "--preonly",
        type=bool,
        default=False,
        help="Specify if preprocessing only (default: False).",
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
    main(preonly=args.preonly, mpnn_type=args.mpnn_type)
