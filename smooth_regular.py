import os
import gc
import random
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils.data as Data

from typing import List
from sklearn.model_selection import KFold
from torch.utils.data.dataset import TensorDataset

from models import *

SEED = 42
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def load_data(del_time: bool, fold: int, num_folds: int = 10, batch_size: int = 128, seed: int = SEED):
    def feat(loc: np.ndarray, vel: np.ndarray) -> np.ndarray:
        return np.concatenate([loc, vel], axis=-1)
    
    def label(edges: np.ndarray, num_atoms: int):
        return np.array((np.reshape(edges, [-1, num_atoms ** 2]) + 1) / 2, dtype=np.int64)

    time_train  = np.load('smooth_spring/time_train.npy')
    time_test   = np.load('smooth_spring/time_test.npy')
    loc_train   = np.load('smooth_spring/loc_train.npy')
    loc_test    = np.load('smooth_spring/loc_test.npy')
    vel_train   = np.load('smooth_spring/vel_train.npy')
    vel_test    = np.load('smooth_spring/vel_test.npy')
    edges_train = np.load('smooth_spring/edges_train.npy')
    edges_test  = np.load('smooth_spring/edges_test.npy')

    if del_time:
        t_length = time_train.shape[1]

        for i in range(1, t_length):
            time_train[:, -i] = time_train[:, -i] - time_train[:, -i-1]
            time_test[:, -i] = time_test[:, -i] - time_test[:, -i-1]
        time_train[:, 0] = 1
        time_test[:, 0] = 1

    num_atoms = int(loc_train.shape[-1] / 2)

    skf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    train_index, val_index = next(itertools.islice(skf.split(loc_train, edges_train), fold, None))

    time_val, loc_val, vel_val, edges_val = time_train[val_index], loc_train[val_index], vel_train[val_index], edges_train[val_index]
    time_train, loc_train, vel_train, edges_train = time_train[train_index], loc_train[train_index], vel_train[train_index], edges_train[train_index]

    loc_max = loc_train.max()
    loc_min = loc_train.min()
    vel_max = vel_train.max()
    vel_min = vel_train.min()

    loc_train = (loc_train - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_train = (vel_train - vel_min) * 2 / (vel_max - vel_min) - 1
    loc_val   = (loc_val - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_val   = (vel_val - vel_min) * 2 / (vel_max - vel_min) - 1
    loc_test  = (loc_test - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_test  = (vel_test - vel_min) * 2 / (vel_max - vel_min) - 1

    feat_train = torch.FloatTensor(feat(loc_train, vel_train))
    edges_train = torch.FloatTensor(label(edges_train, num_atoms))
    feat_val = torch.FloatTensor(feat(loc_val, vel_val))
    edges_val = torch.FloatTensor(label(edges_val, num_atoms))
    feat_test = torch.FloatTensor(feat(loc_test, vel_test))
    edges_test = torch.FloatTensor(label(edges_test, num_atoms))


    off_diag_idx = np.ravel_multi_index(
        np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
        [num_atoms, num_atoms]
    )
    edges_train = edges_train[:, off_diag_idx]
    edges_val = edges_val[:, off_diag_idx]
    edges_test = edges_test[:, off_diag_idx]

    train_data_loader = Data.DataLoader(
        TensorDataset(torch.FloatTensor(time_train), feat_train, edges_train),
        batch_size=batch_size,
        shuffle=True
    )
    val_data_loader = Data.DataLoader(
        TensorDataset(torch.FloatTensor(time_val), feat_val, edges_val),
        batch_size=batch_size
    )
    test_data_loader = Data.DataLoader(
        TensorDataset(torch.FloatTensor(time_test), feat_test, edges_test),
        batch_size=batch_size
    )

    return train_data_loader, val_data_loader, test_data_loader

def seed_all(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

def grid_search(paras: dict) -> List[dict]:
    hypers = list(paras.keys())
    search_table = []
    para_table = [None] * len(hypers)

    for i, para in enumerate(np.meshgrid(*paras.values())):
        para_table[i] = para.flatten()

    for i in range(len(para_table[0])):
        search_table.append({hypers[j]: para_table[j][i] for j in range(len(hypers))})
    return search_table

class EarlyStopping:
    def __init__(self, patience: int = 10, verbose: bool = False, path: str = 'checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = - np.inf
        self.early_stop = False
        self.path = path

    def __call__(self, val_score, model):
        if np.isnan(val_score):
            self.early_stop = True
            return

        if val_score <= self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if self.verbose:
                print(f'Best score increases ({self.best_score:.6f} --> {val_score:.6f}). Saving model...')
            self.best_score = val_score
            torch.save(model, self.path)
            self.counter = 0

def train_model(
    model, model_name: str, save_folder: str,
    train_loader: Data.DataLoader, val_loader: Data.DataLoader,
    num_epochs: int = 200, patience: int = 10,
    device: torch.device = DEVICE,
    lr: float = 0.001, min_lr: float = 0.0001
    ):

    os.makedirs(save_folder, exist_ok=True)
    if model_name.endswith('.pkl'):
        model_name = model_name[:-4]

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs * len(train_loader),
        eta_min=min_lr,
        last_epoch=-1
    )
    early_stopping = EarlyStopping(
        patience=patience,
        verbose=True,
        path=os.path.join(save_folder, f'{model_name}.pkl')
    )

    criterion = nn.BCEWithLogitsLoss()

    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    for epoch in range(num_epochs):
        print(model_name, 'Epoch {}/{}'.format(epoch + 1, num_epochs))
        epoch_loss = 0
        epoch_corrects = 0
        num_sample = 0
        model.train()
        for t, x, y in tqdm(train_loader):
            t, x, y = t.to(device), x.to(device), y.to(device)
            output = model(t, x)
            loss = criterion(output, y)
            loss.backward()

            epoch_corrects += int(torch.sum((output > 0).int() == y))
            epoch_loss += loss.item() * x.size(0)
            num_sample += x.size(0) * y.size(1)

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        train_loss.append(epoch_loss / num_sample)
        train_acc.append(epoch_corrects / num_sample)
        print(' ', train_loss[-1], train_acc[-1])

        epoch_loss = 0
        epoch_corrects = 0
        num_sample = 0
        model.eval()
        with torch.no_grad():
            for t, x, y in tqdm(val_loader):
                t, x, y = t.to(device), x.to(device), y.to(device)
                output = model(t, x)
                loss = criterion(output, y)

                epoch_corrects += int(torch.sum((output > 0).int() == y))
                epoch_loss += loss.item() * x.size(0)
                num_sample += x.size(0) * y.size(1)

        val_loss.append(epoch_loss / num_sample)
        val_acc.append(epoch_corrects / num_sample)
        print(' ', val_loss[-1], val_acc[-1])

        early_stopping(val_acc[-1], model)

        pd.DataFrame({'Train Loss': train_loss, 'Train Acc': train_acc}).to_csv(
            os.path.join(save_folder, f'{model_name}_Train.csv')
        )
        pd.DataFrame({'Val Loss': val_loss, 'Val Acc': val_acc}).to_csv(
            os.path.join(save_folder, f'{model_name}_Val.csv')
        )

        if early_stopping.early_stop:
            print('Early stopping')
            break

    model = torch.load(os.path.join(save_folder, f'{model_name}.pkl'))
    gc.collect()
    return model

def test_model(model, test_loader: Data.DataLoader, device: torch.device = DEVICE) -> float:
    epoch_corrects = 0
    num_sample = 0
    print(' Testing')
    model.eval()
    with torch.no_grad():
        for t, x, y in tqdm(test_loader):
            t, x, y = t.to(device), x.to(device), y.to(device)
            epoch_corrects += int(torch.sum((model(t, x) > 0).int() == y))
            num_sample += x.size(0) * y.size(1)

    return epoch_corrects / num_sample

def main(model_class: nn.Module, base_paras: dict, hyper_paras: List[dict], num_folds: int = 10, num_epochs: int = 200) -> None:
    for para in hyper_paras:
        test_acc = []
        for fold in range(num_folds):
            seed_all(fold)

            model = model_class(**base_paras, **para)
            model_folder = 'Smooth_' + model._get_name()

            train_loader, val_loader, test_loader = load_data(
                True if model_folder.lower().__contains__('ctgru') else False, 
                fold, 
                num_folds
            )

            if len(para) > 1:
                model_name = model_folder + str(tuple(para.values())).replace(', ', '-')
            else:
                model_name = model_folder + '-' + str(tuple(para.values())[0])
            os.makedirs(os.path.join(model_folder, model_name, f'Fold_{fold}'), exist_ok=True)

            model = train_model(
                model, model_name,
                os.path.join(model_folder, model_name, f'Fold_{fold}'),
                train_loader, val_loader,
                num_epochs=num_epochs
            )

            test_acc.append(test_model(model, test_loader))
            print(' Test acc:', test_acc[-1])
            pd.DataFrame(test_acc, columns=[model_name]).to_csv(os.path.join(model_folder, model_name, 'results.csv'))


if __name__ == '__main__':
    main(
        DNS,
        {'input_size': 20, 'output_size': 20},
        grid_search({
            'hidden_size': [128, 256, 512],
            'num_layers': [2],
            'n_blocks': [5, 8, 10]
        }),
        num_folds=5,
        num_epochs=100
    )