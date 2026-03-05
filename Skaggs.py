import seaborn as sb 
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import backend as K
import logging
import argparse

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
            '''if start == 0:
                print("DEBUG out shape:", out.shape, "dtype:", out.dtype)
                print("DEBUG out min/max:", np.min(out), np.max(out))'''

        # out is now (bs, C)
        bs = out.shape[0]
        idx = np.arange(start, start + bs)
        pos_idx = idx // n_rotations   # 0..288

        lam_c_sum += out.sum(axis=0)

        for i in range(bs):
            p = pos_idx[i]
            lam_i_sum[p] += out[i]

    # average over rotations for lam_i, and over all samples for lambda_c
    lam_i = (lam_i_sum / float(n_rotations)).astype(np.float32)                 # (289, C)
    lambda_c = (lam_c_sum / float(image_samples)).astype(np.float32)       # (C,)

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
        print("[Cache] No small cache found. Building it with streaming forward pass...")
        lam_i, lambda_c = cache_builder(
            block=block,
            model=model,
            preprocessed_data=preprocess_funcx,
        )

    occupancy = occupancy_probability(data_path, movement_type='uniform', arena_size=(17,17))
    p = occupancy.reshape(-1)

    '''lambda_from_lam_i = (p[:, None] * lam_i).sum(axis=0)  # (C,)
    print("max abs diff lambda_c vs sum(p*lam_i):", np.max(np.abs(lambda_c - lambda_from_lam_i)))
    print("lambda_c min/max:", lambda_c.min(), lambda_c.max())
    print("lambda_from_lam_i min/max:", lambda_from_lam_i.min(), lambda_from_lam_i.max())'''

    eps = 1e-10
    ratio = lam_i / (lambda_c[None, :] + eps)     # (289, C)
    skaggs = np.sum(p[:, None] * ratio * np.log2(ratio + eps), axis=0).astype(np.float32)  # (C,)

    C = BLOCK_SPECS[block][2]
    return skaggs, lam_i.reshape(17, 17, C) 
    
def find_top_k_skaggs_units(block, skaggs, k=10):
    sorted_skaggs = np.sort(skaggs)[::-1]  
    top_units = sorted_skaggs[:k]  # top k units
    return block, top_units, sorted_skaggs


def build_place_fields():
    sb.heatmap()

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

    
    logging.info("Skaggs analysis completed.")
    occ = occupancy_probability(args.data_path, movement_type='uniform', arena_size=(17,17))
    skaggs_values, skaggs_map = skaggs_list(config['output_layer'],config, model, preprocess_funcx, args.data_path)
    _, top_10_skaggs_units, all_skaggs = find_top_k_skaggs_units('block4_pool', skaggs_values)
    #print("All Skaggs values for block4_pool:", all_skaggs)
    print("Top 10 Skaggs units in block4_pool:", top_10_skaggs_units)
