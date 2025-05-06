## Downloading and Preprocessing the Data

The dataset is available at <https://ogb.stanford.edu/docs/lsc/pcqm4mv2/>. The PyG `pcqm4m-v2` dataset and `pcqm4m-v2-train.sdf` file must be put in the `dataset/raw` directory. The data can be downloaded, preprocessed and pickled on disk using:

```
python pcqm.py --preonly --pickle
```


## Performing HPO

To run the parallel HPO:

```bash
python pcqm_deephyper_parallel_main.py
```

This script internally:

- Defines the search space (hyperparameters).
- Spawns parallel trials using DeepHyper's `MPIWorker` or parallel executor.
- Saves results to a log directory.

---

## ⚙️ Configuration

Edit the following sections in the code to suit your needs:

- **Search space definition** (in `pcqm_deephyper_parallel_main.py`)

    - `["mpnn_type"]`  
      Accepted types: `CGCNN`, `DimeNet`, `EGNN`, `GAT`, `GIN`, `MACE`, `MFC`, `PAINN`, `PNAEq`, `PNAPlus`, `PNA`, `SAGE`, `SchNet` (str)
    - `["num_conv_layers"]`  
      Examples: (`1`, `6`) ... list of (int)
    - `["global_attn_engine"]`
      Accepted types: `GPS`, `None`
    - `["global_attn_heads"]`
      Examples: [`0`, `2`, `4`, `8`] ... list of (int)
    - `["hidden_dim"]`  
      Dimension of node embeddings during convolution (int) - must be a multiple of "global_attn_heads" if "global_attn_engine" is not "None"
    - `["num_headlayers"]`  
      Examples: (`1`, `3`) ... list of (int)
    - `["dim_headlayers"]`  
      Examples: [`64`, `32`] ... list of (int)

- **Trial logic** including model setup, training, and evaluation (in `pcqm_deephyper_parallel_trial.py`)
- **Parallel backend** (MPI, ray, etc. — check DeepHyper documentation)


---

## 📚 References

- [DeepHyper Documentation](https://deephyper.readthedocs.io/)


