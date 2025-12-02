import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score

from utils.utils import *
from utils.pre_processing import *
from utils.nn import *

def get_layers(h_l_n, h_l_s):
    layer_sizes = [h_l_s]
    for _ in range(1, h_l_n):
        layer_sizes.append(layer_sizes[-1] // 2)
    return layer_sizes

def eval_predict_nn(y_real, y_pred, show=True):
    
    accuracy = accuracy_score(y_real, y_pred)
    report = classification_report(y_real, y_pred)
    precision = precision_score(y_real, y_pred)
    recall = recall_score(y_real, y_pred)
    f1 = f1_score(y_real, y_pred)

    if show:
        print(f'Accuracy: {accuracy}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1: {f1}')

    return accuracy, precision, recall, f1

def create_datasets(X_train, y_train, X_test, y_test, X_val, y_val, val=True, batch_size=128):

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

    if val:
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
        val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
    
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)

    if val:
        return train_loader, test_dataset, val_dataset
    else:
        return train_loader, test_dataset

def train_nn(model, train_loader, val_set, optimizer_name='Adam', num_epochs=100, skip=10, patience=10, momentum=0.1):
    
    criterion = model.criterion
    device = model.device
    model.to(device)
    
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=model.lr)
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=model.lr, momentum=momentum)

    history = {'f1_train' : [], 'loss_train': [], 'f1_val': [], 'loss_val': []}
    best_val_loss = float('inf')
    patience_counter = 0

    for e in tqdm(range(1, num_epochs+1)):

        y_hat = np.array([])

        train_epoch_loss = 0
        train_epoch_f1 = 0
        model.train()
        
        for X_train_batch, y_train_batch in train_loader:
            
            y_train_batch = y_train_batch.type(torch.LongTensor)
            X, y = X_train_batch.to(device), y_train_batch.to(device)
            optimizer.zero_grad()
            
            y_pred = model(X)

            y = y.unsqueeze(1)
            loss = criterion(y_pred, y.float())
            
            f1 = f1_score_tensor(y, y_pred)
            
            loss.backward()
            optimizer.step()
            
            train_epoch_loss += loss
            train_epoch_f1 += f1
            y_p = torch.argmax(y_pred, dim=1)
            y_hat = np.concatenate((y_hat, y_p))

        model.eval()
        _, val_loss, val_f1 = evaluate_nn(model, val_set)

        history['f1_train'].append(train_epoch_f1/len(train_loader))
        history['loss_train'].append(train_epoch_loss/len(train_loader))
        history['f1_val'].append(val_f1)
        history['loss_val'].append(val_loss)

        if e%skip == 0:
            print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.3f} | Val Loss: {val_loss:.4f} | Train F1: {train_epoch_f1/len(train_loader):.4f}| Val F1: {val_f1:.4f}')

         # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()  # Save the best model state
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'Early stopping at epoch {e}')
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)  # Load the best model state

    return history, y_hat

def evaluate_nn(model, val_set, metric='f1_score'):

    criterion = model.criterion
    
    X = val_set.tensors[0]
    y = val_set.tensors[1]
    
    with torch.no_grad():
        y_pred = model(X)

    y = y.unsqueeze(1)
    loss = criterion(y_pred, y)
    
    if metric == 'f1_score':
        f1 = f1_score_tensor(y, y_pred)

    y_pred = torch.argmax(y_pred, dim=1)

    return y_pred, loss.item(), f1

def predict_nn(model,test_set,threshold=None):
    
    X = test_set.tensors[0]

    y_pred = model.predict(X,threshold)
    
    return y_pred

def f1_score_tensor(y_true, y_pred):
    
    predictions = torch.round(y_pred)
    
    TP = torch.sum((predictions == 1) & (y_true == 1)).float()
    FP = torch.sum((predictions == 1) & (y_true == 0)).float()
    FN = torch.sum((predictions == 0) & (y_true == 1)).float()
    
    precision = TP / (TP + FP + 1e-10)  # Adding a small epsilon to avoid division by zero
    recall = TP / (TP + FN + 1e-10)
    
    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    return f1.item()