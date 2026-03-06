import seaborn as sb 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import backend as K
import logging
import argparse
from scipy.ndimage import gaussian_filter

import data
import utils
import models

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CACHE_DIR = os.environ.get(
    "CNN_CACHE_DIR",
    os.path.join(SCRIPT_DIR, "cache")
)

os.makedirs(CACHE_DIR, exist_ok=True)

x_min = -1
x_max = 1
z_min = -1
z_max = 1
multiplier = 8

n_rotations = 24

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--config", required=True)
    return p.parse_args()

def cache_builder(block, model, preprocessed_data, n_positions=289):
    cache_path = os.path.join(CACHE_DIR, f"{block}_lam_i.npz")
    batch_size = 24
    BLOCK_SPECS = {
        "block2_pool": (56, 56, 128),
        "block4_pool": (14, 14, 512),
        "block5_pool": (7, 7, 512),
        "fc2": (None, None, 4096)
    }
    if block not in BLOCK_SPECS:
        raise ValueError(f"Unsupported block '{block}' for caching. Supported blocks: {list(BLOCK_SPECS.keys())}")
    
    _, _, C = BLOCK_SPECS[block]
    image_samples = n_positions*n_rotations  # 17*17*24 = 6936

    if preprocessed_data.shape[0] != image_samples:
        raise ValueError(f"Expected {image_samples} samples in preprocessed_data but got {preprocessed_data.shape[0]}.")

    lam_i_sum = np.zeros((n_positions, C), dtype=np.float32)  # (289, C)
    lam_c_sum = np.zeros((C,), dtype=np.float32)  # (C,)
    preprocessed_data = preprocessed_data.astype(np.float32, copy=False)

    for start in range(0, image_samples, batch_size):
        batch = preprocessed_data[start:start+batch_size]
        out = model(batch, training=False).numpy()

        if out.ndim == 4:
            # out: (bs, H, W, C) -> GAP: (bs, C)
            out = out.mean(axis=(1, 2))

        # out is now (bs, C)
        bs = out.shape[0]
        idx = np.arange(start, start + bs)
        pos_idx = idx // n_rotations   # 00000 to 06935, where each position index is repeated 24 times for 24 rotations since that is how the Unity images are organized (all 24 rotations of position 0, then all 24 rotations of position 1, etc.)

        lam_c_sum += out.sum(axis=0) # sum channel activations across all images in batch for lambda_c

        for i in range(bs):
            p = pos_idx[i]
            lam_i_sum[p] += out[i] # add all activations for position p across all rotations

    # average over rotations for lam_i, and over all samples for lambda_c
    lam_i = (lam_i_sum / float(n_rotations)).astype(np.float32) # (289, C), finds average activation for each spatial bin (position)
    lambda_c = (lam_c_sum / float(image_samples)).astype(np.float32) # (C,), average lambda_c across all total images

    # save small cache
    np.savez_compressed(cache_path, lam_i=lam_i, lambda_c=lambda_c)
    print(f"[Cache] Saved small Skaggs cache: {cache_path} (lam_i {lam_i.shape}, lambda_c {lambda_c.shape})")
    return lam_i, lambda_c


def check_data_present(data_path, arena_size=(17,17)):
    if not os.path.exists(data_path):
        return False
    
    H, W = arena_size
    total_expected_images = H * W * n_rotations

    files = [f for f in os.listdir(data_path) if f.endswith('.png')]

    images_present = len(files)
    return images_present == total_expected_images

def occupancy_probability(data_path, movement_type='uniform', arena_size=(17,17)):
    H, W = arena_size
    if movement_type == 'uniform' or movement_type == 'uniform_loc_random_rot':
        if check_data_present(data_path, arena_size):
            occupancy = np.full((H, W), 1/ (H*W), dtype=np.float32)  # initialize with uniform distribution since agent travels to all positions with equal probability
        else:
            raise ValueError(f"Data not present in path {data_path}")
        return occupancy
    else:
        raise ValueError(f"Unsupported movement_type='{movement_type}' in occupancy_probability().")

def skaggs_list(block, config, model, preprocess_funcx, data_path):
    if block == None: 
        raise ValueError("Block cannot be None for Skaggs index calculation.")
    
    BLOCK_SPECS = {
        "block2_pool": (56, 56, 128),
        "block4_pool": (14, 14, 512),
        "block5_pool": (7, 7, 512),
        "fc2": (None, None, 4096)
    }

    if block not in BLOCK_SPECS:
        raise ValueError(f"Unsupported block '{block}' for Skaggs index calculation. Supported blocks: {list(BLOCK_SPECS.keys())}")
    
    cache_path = os.path.join(CACHE_DIR, f"{block}_lam_i.npz")
    if os.path.exists(cache_path):
        print(f"[Cache] Loading small cache: {cache_path}")
        d = np.load(cache_path)
        lam_i = d["lam_i"]
        lambda_c = d["lambda_c"]
    else:
        print("[Cache] No small cache found. Building and caching it now...")
        lam_i, lambda_c = cache_builder(
            block=block,
            model=model,
            preprocessed_data=preprocess_funcx,
        )

    lam_i_smoothed = np.zeros_like(lam_i)

    for c in range(lam_i.shape[1]):
        field = lam_i[:, c].reshape(17,17) # unlfatten to 17x17 spatial map for each unit C. 
        field = gaussian_filter(field, sigma=1) # smooth via Gauss' equation
        lam_i_smoothed[:, c] = field.reshape(-1) # flatten back to (289, C) for Skaggs calculation

    lam_i = lam_i_smoothed

    occupancy = occupancy_probability(data_path, movement_type='uniform', arena_size=(17,17))
    p = occupancy.reshape(-1)

    eps = 1e-10
    ratio = lam_i / (lambda_c[None, :] + eps)
    skaggs = np.sum(p[:, None] * ratio * np.log2(ratio + eps), axis=0).astype(np.float32)  # (C,)

    return skaggs, lam_i 
    
def find_top_k_skaggs_indexes(skaggs, k=10):
    top_k_indexes = np.argsort(skaggs)[::-1][:k]
    return top_k_indexes

def build_place_fields(config, model, preprocess_func, data_path, arena_size=(17,17), k=10):
    block = config['output_layer']
    skaggs, lambda_i = skaggs_list(block=block, config=config, model=model, preprocess_funcx=preprocess_func, data_path=data_path)
    top_k_skaggs_units = find_top_k_skaggs_indexes(skaggs, k=k)

    cols = 5 
    rows = int(np.ceil(k / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4)) # creates figure with subplots in 5x2 shape (for top 10 units), with each subplot being 4x4 inches in size. Adjust cols and rows as needed for different k values.
    axes = np.array(axes).reshape(-1)

    for ax_i, unit in enumerate(top_k_skaggs_units):
        ax = axes[ax_i]
        heatmap = lambda_i[:, unit].reshape(arena_size)
        heatmap = gaussian_filter(heatmap, sigma=1) # apply Gaussian smoothing to the place field for better visualization and so spatial info isn't concentrated and is distributed across neighboring spatial locations, which is more consistent with how place fields are observed in biological neurons. This also helps prevent a single spike in spatial activity from dominating the visualization and allows us to see the overall spatial knowledge of the unit more clearly.
        sb.heatmap(np.log1p(heatmap), ax=ax, cbar=False) # use log scale for better visualization, so large spike in spatial activity doesn't overwhelm neighboring spikes in spatial knowledge

        ax.invert_yaxis()  
        ax.set_title(f"Unit {unit} | Skaggs={skaggs[unit]:.8g}")

    # hide unused axes
    for j in range(len(top_k_skaggs_units), len(axes)):
        axes[j].axis("off")

    #fig.colorbar(im, ax=axes.ravel().tolist())
    fig.suptitle(f"Top {k} place fields in {block} (λ maps)", y=1.02)
    plt.tight_layout()

    makdir = os.path.join(SCRIPT_DIR, "skaggs_place_fields")
    os.makedirs(makdir, exist_ok=True)
    save_path = os.path.join(makdir, f"skaggs_place_fields_{block}.png")
    fig.savefig(save_path)

    plt.show()

    return

if __name__ == "__main__":
    logging.info("Starting Skaggs analysis...")
    args = parse_args()
    config = utils.load_config(args.config)

    preprocess_funcx = data.load_preprocessed_data(
        config = config,
        data_path = args.data_path, 
        movement_mode = '2d',
        env_x_min = x_min,
        env_x_max = x_max,
        env_y_min = z_min,
        env_y_max = z_max,
        multiplier = multiplier,
        n_rotations = n_rotations,
        preprocess_func=None,
        color_mode='rgb', 
        target_size=(224, 224), 
        image_shape=(224, 224, 3),
        interpolation='nearest',
        data_format='channels_last',
        dtype=K.floatx(),
    )

    model = models.load_model(
        model_name=config['model_name'], 
        output_layer=config['output_layer']
    )
    
    if isinstance(model, tuple):
        model = model[0] # unpack model from tuple if necessary

    heat = build_place_fields(config, model, preprocess_funcx, args.data_path, arena_size=(17,17))

    logging.info("Skaggs analysis completed.")