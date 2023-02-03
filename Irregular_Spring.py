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
    train_t  = np.load('irregular_spring/train_t.npy')
    test_t   = np.load('irregular_spring/test_t.npy')
    train_x  = np.load('irregular_spring/train_x.npy')
    test_x   = np.load('irregular_spring/test_x.npy')
    train_y  = np.load('irregular_spring/train_y.npy')
    test_y   = np.load('irregular_spring/test_y.npy')

    if del_time:
        t_length = train_t.shape[1]

        for i in range(1, t_length):
            train_t[:, -i] = train_t[:, -i] - train_t[:, -i-1]
            test_t[:, -i] = test_t[:, -i] - test_t[:, -i-1]
        train_t[:, 0] = 1
        test_t[:, 0] = 1

    skf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    train_index, val_index = next(itertools.islice(skf.split(train_x, train_y), fold, None))

    val_t, val_x, val_y = train_t[val_index], train_x[val_index], train_y[val_index]
    train_t, train_x, train_y = train_t[train_index], train_x[train_index], train_y[train_index]

    train_data_loader = Data.DataLoader(
        TensorDataset(
            torch.FloatTensor(train_t),
            torch.FloatTensor(train_x),
            torch.FloatTensor(train_y)
        ),
        batch_size=batch_size,
        shuffle=True
    )
    val_data_loader = Data.DataLoader(
        TensorDataset(
            torch.FloatTensor(val_t),
            torch.FloatTensor(val_x),
            torch.FloatTensor(val_y)
        ),
        batch_size=batch_size
    )
    test_data_loader = Data.DataLoader(
        TensorDataset(
            torch.FloatTensor(test_t),
            torch.FloatTensor(test_x),
            torch.FloatTensor(test_y)
        ),
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
            model_folder = model._get_name()

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
        pd.DataFrame(test_acc, columns=[model_name]).to_csv(os.path.join(model_folder, model_name, 'results.csv'))


if __name__ == '__main__':
    main(
        CTGRU,
        {'input_size': 20, 'output_size': 20},
        grid_search({
            'hidden_size': [512, 256, 128],
            'tau': [0.5, 1, 2],
            'M': [5, 8]
        }),
        num_folds=5,
        num_epochs=100
    )

    main(
        NeuralCDE,
        {'input_size': 20, 'output_size': 20},
        grid_search({
            'hidden_size': [1024, 512, 256, 128],
            'num_layers': [2, 3, 4],
        }),
        num_folds=5,
        num_epochs=100
    )

    main(
        DNS,
        {'input_size': 20, 'output_size': 20},
        grid_search({
            'hidden_size': [512, 256, 128],
            'num_layers': [2],
            'n_blocks': [5, 8, 10]
        }),
        num_folds=5,
        num_epochs=100
    )

    main(
        DisDNS,
        {'input_size': 20, 'output_size': 20},
        grid_search({
            'hidden_size': [512, 256, 128],
            'num_layers': [2],
            'n_blocks': [5, 8, 10]
        }),
        num_folds=5,
        num_epochs=100
    )