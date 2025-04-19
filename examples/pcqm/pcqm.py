import os
# os.environ["CUDA_VISIBLE_DEVICES"]="4"
import pdb
import json
import torch
import torch_geometric
from torch_geometric.datasets import PCQM4Mv2
from torch_geometric.transforms import AddLaplacianEigenvectorPE
import argparse

# deprecated in torch_geometric 2.0

try:
    from torch_geometric.loader import DataLoader
except ImportError:
    from torch_geometric.data import DataLoader

import hydragnn

from hydragnn.utils.descriptors_and_embeddings.atomicdescriptors import (
    atomicdescriptors,
)

from hydragnn.utils.print.print_utils import print_distributed, iterate_tqdm

from generate_dictionaries_pure_elements import (
    generate_dictionary_elements, generate_ogb_elements
)

# atomicdescriptor = atomicdescriptors(
#     embeddingfilename="./embedding.json",
#     overwritten=True,
#     element_types=generate_ogb_elements(),
#     one_hot=False,
# )

periodic_table = generate_dictionary_elements()

# Update each sample prior to loading.
def pcqm_pre_transform(data, transform):
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


def main(mpnn_type=None, global_attn_engine=None, global_attn_type=None):
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

    # Always initialize for multi-rank training.
    world_size, world_rank = hydragnn.utils.distributed.setup_ddp()

    log_name = f"pcqm_test_{mpnn_type}" if mpnn_type else "pcqm_test"
    # Enable print to log file.
    hydragnn.utils.print.print_utils.setup_log(log_name)

    # LPE
    transform = AddLaplacianEigenvectorPE(
        k=config["NeuralNetwork"]["Architecture"]["pe_dim"],
        attr_name="pe",
        is_undirected=True,
    )

    # Use built-in torch_geometric datasets.
    # Filter function above used to run quick example.
    # NOTE: data is moved to the device in the pre-transform.
    # NOTE: transforms/filters will NOT be re-run unless the pcqm/processed/ directory is removed.
    
    train = PCQM4Mv2(
        root="dataset/pcqm",
        split="train",
        transform=lambda data: pcqm_pre_transform(data, transform),  
    )
    val = PCQM4Mv2(
        root="dataset/pcqm",
        split="val",
        transform=lambda data: pcqm_pre_transform(data, transform),  
    )
    test = PCQM4Mv2(
        root="dataset/pcqm",
        split="test",
        transform=lambda data: pcqm_pre_transform(data, transform), 
    )

    # print_distributed(2, "Add atomic descriptors")
    # train = [
    #     add_atomic_descriptors(data)
    #     for data in iterate_tqdm(train, verbosity_level=2)
    # ]
    # val = [
    #     add_atomic_descriptors(data)
    #     for data in iterate_tqdm(val, verbosity_level=2)
    # ]
    # test = [
    #     add_atomic_descriptors(data)
    #     for data in iterate_tqdm(test, verbosity_level=2)
    # ]

    (train_loader, val_loader, test_loader,) = hydragnn.preprocess.create_dataloaders(
        train, val, test, config["NeuralNetwork"]["Training"]["batch_size"]
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

    # Run training with the given model and pcqm datasets.
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
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the PCQM4Mv2 example with optional model type."
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
    main(mpnn_type=args.mpnn_type)