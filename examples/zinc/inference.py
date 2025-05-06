import os
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

# Import HydraGNN utilities (assuming HydraGNN is installed in the environment)
try:
    import hydragnn
    from hydragnn import utils  # HydraGNN utility module
except ImportError:
    raise ImportError("HydraGNN package is not installed. Please install HydraGNN before running inference.")

def main():
    """Perform inference using a trained HydraGNN model and generate evaluation plots."""
    parser = argparse.ArgumentParser(description="HydraGNN Inference Script")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the trained model checkpoint (file or directory).")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to the model's JSON config file (if not in the same directory as checkpoint).")
    parser.add_argument("--outdir", type=str, default="logs",
                        help="Output directory to save inference plots.")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    config_path = args.config
    out_dir = args.outdir

    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)

    # Determine device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load configuration if provided or if found in checkpoint directory
    config = None
    if config_path:
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
        with open(config_path, "r") as cf:
            config = json.load(cf)
    else:
        # If no config path is given, attempt to find one in the checkpoint directory
        if os.path.isdir(checkpoint_path):
            cand = os.path.join(checkpoint_path, "config.json")
            if os.path.exists(cand):
                with open(cand, "r") as cf:
                    config = json.load(cf)
        else:
            cand = os.path.join(os.path.dirname(checkpoint_path), "config.json")
            if os.path.exists(cand):
                with open(cand, "r") as cf:
                    config = json.load(cf)

    # Initialize distributed processing (DDP) if applicable (this will set up MPI if available)
    try:
        comm_size, rank = utils.setup_ddp()  # Sets up distributed backend, returns (world_size, rank)
    except Exception as e:
        comm_size, rank = 1, 0  # Fallback to single-process if no MPI
    if rank != 0:
        # Only rank 0 will handle plotting to avoid duplication in multi-process scenario
        plt.switch_backend('Agg')  # Use non-interactive backend just in case
    verbosity = 0
    if config and "Verbosity" in config and "level" in config["Verbosity"]:
        verbosity = config["Verbosity"]["level"]

    # Build model architecture
    model = None
    if config:
        # Attempt to construct model according to config (using HydraGNN utilities if available)
        try:
            # If HydraGNN has a helper to initialize model from config:
            model = hydragnn.initialize_model(config)
        except AttributeError:
            # Fallback: HydraGNN might not have initialize_model; try a different approach
            try:
                from hydragnn.utils.model import initialize_model
                model = initialize_model(config)
            except Exception:
                pass
    if model is None:
        # If automatic model creation failed, raise an error for manual model definition
        raise RuntimeError("Failed to initialize model from config. Please ensure the config is correct, "
                           "or manually instantiate the model architecture.")

    # Determine checkpoint file path (if directory given, find a file inside it)
    if os.path.isdir(checkpoint_path):
        # Search for a model file in the directory (commonly with .pt, .pth or .pk extension)
        files = [f for f in os.listdir(checkpoint_path) if f.endswith((".pt", ".pth", ".pk"))]
        if len(files) == 0:
            raise FileNotFoundError(f"No checkpoint file found in directory {checkpoint_path}")
        # If multiple files, pick the most recently modified one
        files.sort(key=lambda f: os.path.getmtime(os.path.join(checkpoint_path, f)), reverse=True)
        checkpoint_file = os.path.join(checkpoint_path, files[0])
        if len(files) > 1:
            print(f"Multiple checkpoint files found. Using '{files[0]}' as the latest checkpoint.")
    else:
        checkpoint_file = checkpoint_path

    # Load model weights from checkpoint
    model_name = os.path.basename(checkpoint_file)
    model_dir = os.path.dirname(checkpoint_file) or "."
    try:
        # Use HydraGNN's load_existing_model if available
        model = hydragnn.load_existing_model(model, model_name, path=model_dir)
        print(f"Loaded model weights from checkpoint: {checkpoint_file}")
    except Exception as e:
        # If HydraGNN utility fails, use direct torch load
        checkpoint_data = torch.load(checkpoint_file, map_location="cpu")
        if isinstance(checkpoint_data, dict):
            # Assume it's a state dict
            model.load_state_dict(checkpoint_data)
            print(f"Loaded model state dict from checkpoint: {checkpoint_file}")
        elif isinstance(checkpoint_data, torch.nn.Module):
            model = checkpoint_data
            print(f"Loaded model object from checkpoint: {checkpoint_file}")
        else:
            raise RuntimeError(f"Unrecognized checkpoint format at {checkpoint_file}")

    # Move model to appropriate device and wrap in DDP if applicable
    model = utils.get_distributed_model(model, verbosity=verbosity)
    model.to(device)
    model.eval()

    # -----------------------------------------------------------------------
    # TODO: Load your dataset here and prepare a DataLoader for inference.
    # For example, if using PyTorch Geometric:
    # from torch_geometric.data import DataLoader
    # test_dataset = ...  # create or load dataset
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    # Ensure that each data object contains the ground truth (e.g., in data.y).
    # -----------------------------------------------------------------------
    test_loader = None  # Placeholder: replace with actual DataLoader

    if test_loader is None:
        raise RuntimeError("No test data provided. Please add your data loading logic where indicated.")

    # Lists to collect predictions and ground truths
    all_preds = []
    all_truths = []

    # Perform inference on the test dataset
    with torch.no_grad():
        for data in test_loader:
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

        # Plot Prediction vs Ground Truth
        plt.figure()
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
            plt.title(f"Predictions - {output_names[i]}")
            plt.xlabel("Sample index")
            plt.ylabel(f"Predicted {output_names[i]}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"pred_vs_true_{safe_names[i]}.png"))
        plt.close()

        # Plot Residual distribution (if ground truth is available)
        if true_flat is not None and true_flat.size > 0:
            residuals = pred_flat - true_flat
            plt.figure()
            plt.hist(residuals, bins=50, color='gray', edgecolor='black')
            plt.axvline(x=0.0, color='k', linestyle='--')  # reference line at zero error
            plt.xlabel(f"Residuals for {output_names[i]} (Pred - True)")
            plt.ylabel("Frequency")
            mu = np.nanmean(residuals)
            sigma = np.nanstd(residuals)
            plt.title(f"Residual Distribution - {output_names[i]}\nMean = {mu:.4f}, Std = {sigma:.4f}")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"residuals_{safe_names[i]}.png"))
            plt.close()

            # Compute and print MAE for this output
            mae = np.nanmean(np.abs(residuals))
            print(f"{output_names[i]} - Mean Absolute Error (MAE): {mae:.4f}")
        else:
            print(f"{output_names[i]} - No ground truth available, skipped residual plot.")

    if rank == 0:
        print(f"Inference complete. Plots saved to '{out_dir}' directory.")

if __name__ == "__main__":
    main()
