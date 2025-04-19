import os, sys, json, pdb
import logging
import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

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
    generate_dictionary_elements,
)

def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))

#torch.backends.cudnn.enabled = False

def reverse_dict(input_dict):
    """Reverses a dictionary, swapping keys and values."""
    return {value: key for key, value in input_dict.items()}

# FIXME: this radis cutoff overwrites the radius cutoff currently written in the JSON file
create_graph_fromXYZ = RadiusGraph(r=5.0)  # radius cutoff in angstrom
compute_edge_lengths = Distance(norm=False, cat=True)

#atomicdescriptor = atomicdescriptors(
#    embeddingfilename="./embedding.json",
#    overwritten=True,
#    element_types=None,
#    one_hot=False,
#)

periodic_table = generate_dictionary_elements()
reverse_pt = reverse_dict(periodic_table)

def get_atomic_number(symbol):
    return reverse_pt.get(symbol)

# Update each sample prior to loading.
def tmqm_pre_transform(data, transform):
    # LPE
    data = transform(data)

    # gps requires relative edge features, introduced rel_lapPe as edge encodings
    source_pe = data.pe[data.edge_index[0]]
    target_pe = data.pe[data.edge_index[1]]
    data.rel_pe = torch.abs(source_pe - target_pe)  # Compute feature-wise difference
    return data

#def add_atomic_descriptors(data):
#    descriptor_tensor = torch.empty((data.x.shape[0], 18))
#    for atom_id in range(data.x.shape[0]):
#        atomic_string = periodic_table[int(data.x[atom_id, 0].item())]
#        descriptor_tensor[atom_id, :] = atomicdescriptor.get_atom_features(
#            atomic_string
#        )
#        data.x = torch.cat([data.x, descriptor_tensor], dim=1)

#    return data

class tmQM(AbstractBaseDataset):
    def __init__(
        self, datadir, pe_dim
    ):
        super().__init__()
        
        datafiles = datadir+'/tmQM_'

        # Read atom_type, coordinates and properties (molecule_charge,spin,metal_node_degree,stoichiometry)
        xyz = []
        for i in range(3):
            with open(datafiles+'X'+str(i+1)+'.xyz') as file:
                for line in file:
                    xyz.append(line.rstrip())

        # Read atomic valence indices
        bo = []
        for i in range(3):
            with open(datafiles+'X'+str(i+1)+'.BO') as file:
                for line in file:
                    bo.append(line.rstrip())

        # Read remaining properties (electronic_energy,dispersion_energy,dipole_moment,
        #natural_charge_at_metal_center,homolumo_gap,homo_energy,lumo_energy,polarizability)
        y = pd.read_csv(datafiles+'y.csv',sep=';')
        groundtruths = y.columns.tolist()[1:-1]

        # Read charges on atoms
        q = []
        with open(datafiles+'X.q') as file:
            for line in file:
                q.append(line.rstrip())


        # LPE
        transform = AddLaplacianEigenvectorPE(
            k=pe_dim,
            attr_name="pe",
            is_undirected=True,
        )

        size = len(xyz)
        counter = 1
        bo_count = 1

        pbar = tqdm(total=108541)
        while counter < size:
            # Create the PyTorch Geometric Data object
            data = Data()

            # Number of atoms in molecule            
            n_lines = int(xyz[counter-1])
            atom_count = 0
            if counter == 180009: # Skip duplicate data entry
                bo_count += 26
            
            # Generate node features (x) and (pos)
            x = np.zeros((n_lines,3))
            pos = np.zeros((n_lines,3))
            j = 0
            for i in range(counter+1, counter+n_lines+1):
                line = xyz[i].split()
                x[j][0] = float(get_atomic_number(line[0]))
                x[j][1] = float(q[i-1].split()[-1])
                x[j][2] = float(bo[bo_count].split()[2])    
                pos[j] = np.array([float(line[1]),float(line[2]),float(line[3])])   
                j += 1
                atom_count += 1
                bo_count += 1
            bo_count += 2

            data.x = torch.Tensor(x)
            data.pos = torch.Tensor(pos)

            # Create radius graph
            data = create_graph_fromXYZ(data)
            # Add edge length as edge feature
            data = compute_edge_lengths(data)
            data.edge_attr = data.edge_attr.to(torch.float32)

            # Generate targets
            tar = np.zeros(11)
            descriptors = xyz[counter].replace(" ", "").split('|')
            data[descriptors[3].split('=')[0]] = descriptors[3].split('=')[1]
            data[descriptors[0].split('=')[0]] = descriptors[0].split('=')[1]
            tar[0] = float(descriptors[1].split('=')[1])
            tar[1] = float(descriptors[2].split('=')[1])   
            tar[2] = float(descriptors[4].split('=')[1])
            gt_vals = y.loc[y['CSD_code']==data['CSD_code']].values.tolist()[0][1:-1]
            tar[3:] = gt_vals
            data.y = torch.Tensor(tar).unsqueeze(1)

            # Delete non-array fields for adios (only for frontier)
            del(data['CSD_code'])
            del(data['Stoichiometry'])

            # Encoders
            data = tmqm_pre_transform(data, transform)
            #data = add_atomic_descriptors(data)

            self.dataset.append(data)
            counter += (n_lines + 3)
            pbar.update(1)
        pbar.close()

        random.shuffle(self.dataset)

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
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmqm.json")
    with open(filename, "r") as f:
        config = json.load(f)

    ##################################################################################################################
    # Always initialize for multi-rank training.
    comm_size, rank = hydragnn.utils.distributed.setup_ddp()
    ##################################################################################################################

    comm = MPI.COMM_WORLD

    modelname = "tmqm" 

    if preonly:
        ## local data
        dataset = tmQM(
            'dataset/raw', config["NeuralNetwork"]["Architecture"]["pe_dim"],
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

    log_name = f"tmqm_test_{mpnn_type}" if mpnn_type else "tmqm_test"
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

    if ddstore:
        os.environ["HYDRAGNN_AGGR_BACKEND"] = "mpi"
        os.environ["HYDRAGNN_USE_ddstore"] = "1"

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
        description="Run the tmQM example with optional model type."
    )
    parser.add_argument(
        "--preonly",
        type=bool,
        default=False,
        help="Specify if preprocessing only (default: False).",
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
    parser.set_defaults(format="adios")
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
