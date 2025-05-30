import os,sys
import pdb
import json
import logging
import tarfile
import argparse
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from rdkit import Chem
from mpi4py import MPI
from sklearn.model_selection import train_test_split

import pyarrow as pa
import pyarrow.parquet as pq

from ogb.lsc import PCQM4Mv2Dataset
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector

import random
import torch
# torch.cuda.init()
# from mpi4py import MPI
# FIX random seed
random_state = 0
torch.manual_seed(random_state)
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.transforms import AddLaplacianEigenvectorPE
from torch_geometric.utils import degree

# deprecated in torch_geometric 2.0
try:
    from torch_geometric.loader import DataLoader
except ImportError:
    from torch_geometric.data import DataLoader

import hydragnn
from hydragnn.utils.profiling_and_tracing.time_utils import Timer
from hydragnn.utils.model import print_model
from hydragnn.utils.descriptors_and_embeddings.topologicaldescriptors import compute_topo_features
from hydragnn.utils.descriptors_and_embeddings.atomicdescriptors import (
    atomicdescriptors,
)
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
    generate_dictionary_elements, generate_ogb_elements
)

def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))

def reverse_dict(input_dict):
    """Reverses a dictionary, swapping keys and values."""
    return {value: key for key, value in input_dict.items()}

def has_isolated_nodes(data: Data) -> bool:
    # compute degree of each node
    deg = degree(data.edge_index[0], num_nodes=data.num_nodes)
    # if any degree == 0, there's an isolated node
    return bool((deg == 0).any())

periodic_table = generate_dictionary_elements()

class pcqm(AbstractBaseDataset):
    def __init__(
        self, datadir, num_laplacian_eigs
    ):
        super().__init__()

        # URL of the file to download
        self.URL = 'http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2-train.sdf.tar.gz'
        # Path to save downloaded file
        self.TAR_FILE_NAME = 'pcqm4m-v2-train.sdf.tar.gz'
        self.SDF_FILE_NAME = 'pcqm4m-v2-train.sdf'

        os.makedirs(datadir, exist_ok=True)

        if os.path.exists(datadir+'records.parquet'):
            print('Found records')
            records = pd.read_parquet(datadir+'records.parquet')
        else:
            print('Creating records')
            sdf_file = self.download_and_extract_sdf(root=datadir)
            records = self.process_dataset(sdf_file, root=datadir)

        # LPE
        transform = AddLaplacianEigenvectorPE(
            k=num_laplacian_eigs,
            attr_name="pe",
            is_undirected=True,
        )

        pbar = tqdm(total=records.shape[0],desc="Pre-processing data")
        for idx, row in records.iterrows():
           # Create the PyTorch Geometric Data object
            data = Data()

            data.x = torch.from_numpy(row['node_features']).reshape([-1,9]).to(torch.float32)
            data.pos = torch.from_numpy(row['dft_coords']).reshape([-1,3]).to(torch.float32)
            data.edge_index = torch.from_numpy(row['edges']).reshape([-1,2]).to(torch.int64).transpose(1,0)
            data.edge_attr = torch.from_numpy(row['edge_features']).reshape([-1,3]).to(torch.float32)
            data.y = torch.Tensor([row['target']]).to(torch.float32)
            
            if not has_isolated_nodes(data):
                # Pre-transform
                try:
                    data = transform(data)
                    self.dataset.append(data)
                except:
                    print("Laplacian_eigs do not converge for graph {} ".format(idx))
            else:
                print("Graph {} has one or more isolated nodes".format(idx))
            pbar.update(1)
        pbar.close()

        self.get_topo_encodings()

        random.shuffle(self.dataset)

    def get_topo_encodings(self):
        n_procs = min(cpu_count(), len(self.dataset))
        chunksize = max(1, len(self.dataset) // (n_procs * 4))  # tune this

        with Pool(processes=n_procs) as pool:
            iterator = pool.imap(compute_topo_features, self.dataset, chunksize)
            self.dataset = list(tqdm(iterator, total=len(self.dataset), desc="Adding topological encodings"))
            # self.dataset = pool.map(compute_topo_features, self.dataset, chunksize)

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        return self.dataset[idx]

    def download_sdf_tar(self, download_path: str) -> str:
        """
        Download the SDF tar.gz file from the URL.

        Args:
            download_path (str): The path to save the downloaded file.

        Returns:
            str: The path to the downloaded tar file.
        """
        tar_path = os.path.join(download_path, self.TAR_FILE_NAME)

        # Download the file with progress bar
        print(f"Downloading SDF file from {self.URL}...")
        if os.path.exists(tar_path):
            print(f"File already exists at {tar_path}. Skipping download.")
            return tar_path

        response = requests.get(self.URL, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 KB

        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
        try:
            with open(tar_path, 'wb') as f:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    f.write(data)
        finally:
            progress_bar.close()

        print("Download complete.")
        return tar_path


    def extract_tar(self, tar_path: str, output_dir: str) -> str:
        """
        Extract the SDF file from the downloaded tar.gz file.

        Args:
            tar_path (str): The path to the tar file.
            output_dir (str): The directory to extract the SDF file.

        Returns:
            str: The path to the extracted SDF file.
        """
        sdf_path = os.path.join(output_dir, self.SDF_FILE_NAME)

        # Extract the downloaded tar.gz file
        print("Extracting tar file...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extract(self.SDF_FILE_NAME, path=output_dir, filter='data')
        print("Extraction complete.")

        return sdf_path


    def download_and_extract_sdf(self, root: str) -> str:
        """
        Download and extract the SDF file.

        Args:
            root (str): The root directory to save the files.

        Returns:
            str: The path to the extracted SDF file.
        """
        tar_path = os.path.join(root, self.TAR_FILE_NAME)
        sdf_path = os.path.join(root, self.SDF_FILE_NAME)

        if not os.path.exists(sdf_path):
            if not os.path.exists(tar_path):
                download_path = self.download_sdf_tar(download_path=root)
            else:
                print("Using existing tar file...")
                download_path = tar_path

            sdf_path = self.extract_tar(download_path, output_dir=root)
        else:
            print("Using existing SDF file...")

        return sdf_path


    def mol2graph(self, mol: Chem.Mol) -> tuple:
        """
        Convert an RDKit molecule object to a graph representation.

        Args:
            mol (Chem.Mol): The RDKit molecule object.

        Returns:
            tuple: A tuple containing:
                - num_nodes (np.int16): The number of nodes in the graph.
                - edges (np.ndarray): The edges of the graph represented as an array of shape (2, num_edges).
                - node_features (np.ndarray): The node features represented as an array of shape (num_nodes, num_node_features).
                - edge_features (np.ndarray): The edge features represented as an array of shape (num_edges, num_edge_features).
        """
        # atoms
        atom_features_list = []
        for atom in mol.GetAtoms():
            atom_features_list.append(atom_to_feature_vector(atom))
        x = np.array(atom_features_list, dtype=np.int64)

        # bonds
        num_bond_features = 3  # bond type, bond stereo, is_conjugated
        if len(mol.GetBonds()) > 0:  # mol has bonds
            edges_list = []
            edge_features_list = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                edge_feature = bond_to_feature_vector(bond)

                # add edges in both directions
                edges_list.append((i, j))
                edge_features_list.append(edge_feature)
                edges_list.append((j, i))
                edge_features_list.append(edge_feature)

            # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
            edge_index = np.array(edges_list, dtype=np.int64).T

            # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
            edge_attr = np.array(edge_features_list, dtype=np.int64)

        else:  # mol has no bonds
            edge_index = np.empty((2, 0), dtype=np.int64)
            edge_attr = np.empty((0, num_bond_features), dtype=np.int64)

        num_nodes = np.array(len(x), dtype=np.int16)
        edges = edge_index.T.astype(np.int16)
        edge_features = edge_attr.astype(np.int16)
        node_features = x.astype(np.int16)

        return num_nodes, edges, node_features, edge_features


    def process_sdf(self, sdf_file: str, remove_sdf: bool = True) -> dict:
        """
        Process the SDF file and extract relevant information.

        Args:
            sdf_file (str): The path to the SDF file.
            remove_sdf (bool, optional): Whether to remove the SDF file after processing. Defaults to True.

        Returns:
            dict: A dictionary containing the processed data.
        """
        print('Opening SDF ...')
        suppl = Chem.SDMolSupplier(sdf_file)

        records = {
            'idx': [],
            'num_nodes': [],
            'edges': [],
            'node_features': [],
            'edge_features': [],
            'dft_coords': []
        }

        print('Processing SDF ...')
        for idx, mol in enumerate(tqdm(suppl)):
            mol = Chem.RemoveAllHs(mol)
            num_nodes, edges, node_features, edge_features = self.mol2graph(mol)
            dft_coords = mol.GetConformer().GetPositions().astype('float32')

            records['idx'].append(idx)
            records['num_nodes'].append(num_nodes)
            records['edges'].append(edges.ravel())
            records['node_features'].append(node_features.ravel())
            records['edge_features'].append(edge_features.ravel())
            records['dft_coords'].append(dft_coords.ravel())

        return records


    def process_dataset(self, sdf_file: str, root: str) -> tuple:
        """
        Process the PCQM4Mv2 dataset.

        Args:
            sdf_file (str): The path to the SDF file.
            root (str): The root directory of the dataset.
            remove_sdf (bool, optional): Whether to remove the SDF file after processing. Defaults to False.

        Returns:
            tuple: A tuple containing:
                - records (dict): A dictionary containing the processed data.
                - splits (dict): A dictionary containing the dataset splits.
        """
        print('Extracting training data from SDF before other splits')
        records = self.process_sdf(sdf_file)

        print('Loading PCQM4Mv2 dataset')
        dataset = PCQM4Mv2Dataset(root=root, only_smiles=True)
        train_idx, val_idx, test_idx = (dataset.get_idx_split()[k]
                                        for k in ['train', 'valid', 'test-dev'])
        assert np.all(train_idx == records['idx'])

        print('Adding training targets from PCQM4Mv2 dataset')
        records['target'] = []
        for idx in tqdm(train_idx):
            _, target = dataset[idx]
            target = np.array(target, dtype=np.float32)
            records['target'].append(target)

        # def process_split(split: str, split_idx: np.ndarray):
        #     print(f'Processing {split} data from PCQM4Mv2 dataset')
        #     for idx in tqdm(split_idx):
        #         smiles, target = dataset[idx]
        #         target = np.array(target, dtype=np.float32)

        #         mol = Chem.MolFromSmiles(smiles)
        #         num_nodes, edges, node_features, edge_features = self.mol2graph(mol)

        #         records['idx'].append(idx)
        #         records['num_nodes'].append(num_nodes)
        #         records['edges'].append(edges.ravel())
        #         records['node_features'].append(node_features.ravel())
        #         records['edge_features'].append(edge_features.ravel())
        #         records['target'].append(target)

        # process_split('valid', val_idx)
        # process_split('test', test_idx)

        records['idx'] = np.stack(records['idx'])
        records['num_nodes'] = np.stack(records['num_nodes'])
        records['target'] = np.stack(records['target'])

        records_dict = {
            'idx': records['idx'],
            'num_nodes': records['num_nodes'],
            'edges': records['edges'],
            'node_features': records['node_features'],
            'edge_features': records['edge_features'],
            'dft_coords': records['dft_coords'],
            'target': records['target'],
        }

        pq.write_table(pa.table(records_dict), root+'records.parquet')

        return pd.DataFrame.from_dict(records_dict)

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
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pcqm.json")
    with open(filename, "r") as f:
        config = json.load(f)

    ##################################################################################################################
    # Always initialize for multi-rank training.
    comm_size, rank = hydragnn.utils.distributed.setup_ddp()
    ##################################################################################################################

    comm = MPI.COMM_WORLD

    modelname = "pcqm" 

    if preonly:
        ## local data
        dataset = pcqm(
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

    log_name = f"pcqm_test_{mpnn_type}" if mpnn_type else "pcqm_test"
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
        create_plots=True
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the pcqm example with optional model type."
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
