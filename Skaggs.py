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
    if movement_type == 'uniform':
        if check_data_present(data_path, arena_size):
            occupancy = np.full((H, W), 1/ (H*W), dtype=np.float32)  # initialize with uniform distribution since agent travels to all positions with equal probability
        else:
            raise ValueError(f"Data not present in path {data_path}")
        return occupancy
    else:
        raise ValueError(f"Unsupported movement_type='{movement_type}' in occupancy_probability().")

def mean_firing_rate(config, model, preprocess_data):
    """Returns mean firing rate per channel across all spatial positions and rotations."""
    model_reps = data.load_full_dataset_model_reps(config, model, preprocess_data)

    reps_array = model_reps.reshape(6936, 56, 56, 128)
    mean_per_channel = reps_array.mean(axis=(0,1,2))  # (128,)

    return mean_per_channel

def skaggs(config, model, preprocess_funcx, data_path):
    lambda_channel = mean_firing_rate(config, model, preprocess_funcx)
    occupancy = occupancy_probability(data_path, movement_type='uniform', arena_size=(17,17))
    pass

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

    mfr = mean_firing_rate(
        config = config,
        model = model, 
        preprocess_data = preprocess_funcx
        )
    
    print("mfr shape:", mfr.shape)
    print("mfr first 10:", mfr[:10])
    logging.info("Skaggs analysis completed.")