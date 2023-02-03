import os
import random
import numpy as np

from tqdm import tqdm

def irregularize_data(p: float = 0.5):
    loc_train = np.load('spring/loc_train.npy')
    loc_test = np.load('spring/loc_test.npy')
    vel_train = np.load('spring/vel_train.npy')
    vel_test = np.load('spring/vel_test.npy')
    edges_train = np.load('spring/edges_train.npy')
    edges_test = np.load('spring/edges_test.npy')

    # [num_samples, num_timesteps, num_dims, num_atoms]
    num_atoms = loc_train.shape[3]

    loc_max = loc_train.max()
    loc_min = loc_train.min()
    vel_max = vel_train.max()
    vel_min = vel_train.min()

    # Normalize to [-1, 1]
    loc_train = (loc_train - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_train = (vel_train - vel_min) * 2 / (vel_max - vel_min) - 1

    loc_test = (loc_test - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_test = (vel_test - vel_min) * 2 / (vel_max - vel_min) - 1

    loc_train = loc_train.reshape(*loc_train.shape[:2], -1)
    vel_train = vel_train.reshape(*vel_train.shape[:2], -1)

    feat_train = np.concatenate([loc_train, vel_train], axis=-1)
    edges_train = np.reshape(edges_train, [-1, num_atoms ** 2])
    edges_train = np.array((edges_train + 1) / 2, dtype=np.int64)

    loc_test = loc_test.reshape(*loc_test.shape[:2], -1)
    vel_test = vel_test.reshape(*vel_test.shape[:2], -1)

    feat_test = np.concatenate([loc_test, vel_test], axis=-1)
    edges_test = np.reshape(edges_test, [-1, num_atoms ** 2])
    edges_test = np.array((edges_test + 1) / 2, dtype=np.int64)

    feat_train_, t_train, feat_test_, t_test = [], [], [], []

    max_length = feat_train.shape[1]
    n_taken = int(p * max_length)
    
    print('Generating Training Set')
    for i in tqdm(range(loc_train.shape[0])):
        choices = np.sort(np.random.choice(np.arange(1, max_length - 1), size=n_taken - 2, replace=False))
        choices = np.append(np.array(0), choices)
        choices = np.append(choices, np.array(max_length - 1))
        feat_train_.append(feat_train[i][choices])
        t_train.append(choices)

    print('Generating Test Set')
    for i in tqdm(range(loc_test.shape[0])):
        choices = np.sort(np.random.choice(np.arange(1, max_length - 1), size=n_taken - 2, replace=False))
        choices = np.append(np.array(0), choices)
        choices = np.append(choices, np.array(max_length - 1))
        feat_test_.append(feat_test[i][choices])
        t_test.append(choices)

    off_diag_idx = np.ravel_multi_index(
        np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
        [num_atoms, num_atoms])
    edges_train = edges_train[:, off_diag_idx]
    edges_test = edges_test[:, off_diag_idx]
    
    print(np.stack(feat_train_).shape)
    print(np.stack(t_train).shape)

    print(np.stack(feat_test_).shape)
    print(np.stack(t_test).shape)

    os.makedirs('irregular_spring', exist_ok=True)
    np.save('irregular_spring/train_x', np.stack(feat_train_))
    np.save('irregular_spring/train_t', np.stack(t_train))
    np.save('irregular_spring/train_y', np.stack(edges_train))

    np.save('irregular_spring/test_x', np.stack(feat_test_))
    np.save('irregular_spring/test_t', np.stack(t_test))
    np.save('irregular_spring/test_y', np.stack(edges_test))

    print('done')

seed = 42
random.seed(seed)
np.random.seed(seed)
irregularize_data(p=0.4)