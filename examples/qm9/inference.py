import os, json, pdb
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
from scipy.stats import pearsonr

import torch
# torch.cuda.init()
# from mpi4py import MPI
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

def main(out_dir, checkpoint_path, format='pickle', ddstore=False, 
        ddstore_width=None, shmem=False):
    
    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)

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
    filename = os.path.join(checkpoint_path+"/config.json")
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

    # log_name = f"tmqm_test_{mpnn_type}" if mpnn_type else "tmqm_test"
    # # Enable print to log file.
    # hydragnn.utils.print.print_utils.setup_log(log_name)

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

    # # Print details of neural network architecture
    # print_model(model)

    # Load model weights from checkpoint
    model = load_existing_model(model, path=checkpoint_path)

    # Inference mode
    model.eval()

    # Lists to collect predictions and ground truths
    all_preds = []
    all_truths = []

    # Perform inference on the test dataset
    with torch.no_grad():
        for data in tqdm(test_loader):
            # Move data to device (handles torch_geometric Data objects)
            data = data.to(device)
            # Forward pass to get model predictions
            output = model(data)
            # Detach and move predictions to CPU for analysis
            if isinstance(output, (list, tuple)):
                # Model returns multiple outputs (list/tuple), e.g., for multi-task
                outputs = [out.detach().cpu() for out in output]
            else:
                outputs = [output.detach().cpu()]
            # Gather ground truth values from data
            if hasattr(data, "y") and data.y is not None:
                true_vals = data.y.detach().cpu()
            else:
                # If ground truth is stored in a custom attribute, handle accordingly.
                true_vals = torch.tensor([])  # empty if not available
            all_preds.append(outputs)
            all_truths.append(true_vals)

    # If no data processed (should not happen if test_loader is correctly set)
    if len(all_preds) == 0:
        print("Warning: No data processed. Check if test_loader is properly loaded with data.")
        return

    # Determine output names from config if available
    output_names = []
    if config:
        voi = config.get("NeuralNetwork", {}).get("Variables_of_interest", {})
        if "output_names" in voi:
            output_names = voi["output_names"]
    # Infer number of outputs from model predictions
    num_outputs = len(all_preds[0]) if isinstance(all_preds[0], list) else 1
    if not output_names or len(output_names) != num_outputs:
        output_names = [f"Output_{i}" for i in range(num_outputs)]

    # Concatenate predictions and truths across all batches
    # all_preds is a list of lists: outer list over batches, inner list over outputs
    preds_by_output = [ [] for _ in range(num_outputs) ]
    for batch_outputs in all_preds:
        for i, out_tensor in enumerate(batch_outputs):
            preds_by_output[i].append(out_tensor)
    preds_by_output = [ torch.cat(outputs_list) for outputs_list in preds_by_output ]
    # Concatenate all ground truth values (assuming all_truths are graph-level targets in data.y)
    true_all = torch.cat(all_truths) if all_truths else torch.tensor([])

    # If ground truth tensor has multiple columns (multi-target global outputs), split by column
    truths_by_output = []
    if true_all.ndim > 1 and true_all.shape[1] > 1:
        # Assume each column corresponds to a distinct output property
        for j in range(true_all.shape[1]):
            truths_by_output.append(true_all[:, j])
    else:
        truths_by_output.append(true_all.flatten())

    # Ensure the number of output lists matches (in case some outputs are node-level without direct single tensor ground truth)
    if len(truths_by_output) != num_outputs:
        # If mismatch, attempt to align by using available ground truth for global outputs
        # For simplicity, extend or truncate truths_by_output to match preds_by_output count
        if len(truths_by_output) < num_outputs:
            # Pad missing ground truths with NaNs for plotting (if node-level outputs have no direct truth here)
            for _ in range(num_outputs - len(truths_by_output)):
                truths_by_output.append(torch.full_like(truths_by_output[0], float('nan')))
        else:
            truths_by_output = truths_by_output[:num_outputs]

    # Convert output names to safe strings for filenames
    safe_names = [ "".join([c if c.isalnum() or c in "-_." else "_" for c in name]) for name in output_names ]

    # Generate plots for each output
    for i in range(num_outputs):
        pred_vals = preds_by_output[i].numpy()
        true_vals = truths_by_output[i].numpy() if i < len(truths_by_output) else None

        # Flatten values (covers cases with vector outputs or node-level outputs where each entry is a scalar point)
        pred_flat = pred_vals.flatten()
        true_flat = true_vals.flatten() if true_vals is not None else None

        # Compute MSE, MAE, pearson correlation for this output
        mae = np.nanmean(np.abs(pred_flat - true_flat))
        mse = np.nanmean(np.square(pred_flat - true_flat))
        correlation, p_value = pearsonr(true_flat, pred_flat)
        # print(f"{output_names[i]} - Mean Absolute Error (MAE): {mae:.4f}")

        # Plot Prediction vs Ground Truth
        plt.figure(figsize=(10, 8))
        if true_flat is not None:
            plt.scatter(true_flat, pred_flat, alpha=0.7, edgecolors='none')
            min_val = np.nanmin(true_flat)
            max_val = np.nanmax(true_flat)
            # Extend range a bit for the identity line
            pad = 0.05 * (max_val - min_val) if max_val != min_val else 1.0
            line_min = min_val - pad
            line_max = max_val + pad
            plt.plot([line_min, line_max], [line_min, line_max], 'k--', lw=2)
            plt.xlabel(f"True {output_names[i]}")
            plt.ylabel(f"Predicted {output_names[i]}")
            plt.title(f"Predicted vs True - {output_names[i]}")
        else:
            # If no ground truth available, just plot predictions
            plt.plot(pred_flat, label="Predictions")
            plt.title(f"Prediction Plot \nMSE = {mse:.4f}, MAE = {mae:.4f} \npearson_corr = {correlation:.4f}, pval = {p_value:.4f} ")
            plt.xlabel("Sample index")
            plt.ylabel(f"Predicted {output_names[i]}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"pred_vs_true_{safe_names[i]}.png"))
        plt.close()

        # Plot Residual distribution (if ground truth is available)
        if true_flat is not None and true_flat.size > 0:
            residuals = pred_flat - true_flat
            plt.figure(figsize=(10, 8))
            plt.hist(residuals, bins=50, color='gray', edgecolor='black')
            plt.axvline(x=0.0, color='k', linestyle='--')  # reference line at zero error
            plt.xlabel(f"Residuals for {output_names[i]} (Pred - True)")
            plt.ylabel("Frequency")
            mu = np.nanmean(residuals)
            sigma = np.nanstd(residuals)
            plt.title(f"Residual Distribution \nMean = {mu:.4f}, Std = {sigma:.4f}")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"residuals_{safe_names[i]}.png"))
            plt.close()
        else:
            print(f"{output_names[i]} - No ground truth available, skipped residual plot.")

    if rank == 0:
        print(f"Inference complete. Plots saved to '{out_dir}' directory.")

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

    out_dir = 'results/best_model_wgps'
    checkpoint_path = 'results/HPO_wgps/logs/tmqm_hpo_trials_0.52'

    main(out_dir, checkpoint_path, format=args.format, ddstore=args.ddstore, 
        ddstore_width=args.ddstore_width, shmem=args.shmem)
