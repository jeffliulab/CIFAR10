# ALL CODES DETAILS ARE IN trainer.py FOR BETTER MODULARITY

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import time

from datasets.cifar10 import get_loaders
from models.vgg13 import VGG13 # 牛津大学的CNN架构Visual Geometry Group（VGG）
# VGG13有13个卷积层和3个全连接层，适用于CIFAR-10数据集
from train.checkpoint import Checkpointer

def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Train `model` for a single epoch.

    Args:
        model:    the neural network being trained
        loader:    DataLoader that yields (inputs, labels) mini-batches
        criterion:  ls function (e.g., CrossEntropyLoss)
        optimizer:  optimizer instance (e.g., SGD/Adam)
        device:    torch.device("cuda") for GPUs or torch.device("cpu") for CPUs
    Returns:
        avg_loss:   average training loss over all samples in this epoch
        acc:    training accuracy (%) over all samples in this epoch
    """
    model.train()
    running_loss = 0.0
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * x.size(0)
        preds = pred.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return running_loss / total, 100 * correct / total
    

def eval_one_epoch(model, loader, criterion, device):
    """
    Evaluate `model` for a single epoch on the validation/test dataset.
    
    Args:
        model:    the neural network being trained
        loader:    DataLoader that yields (inputs, labels) mini-batches
        criterion:  loss function (e.g., CrossEntropyLoss)
        device:    torch.device("cuda") for GPUs or torch.device("cpu") for CPUs

    Returns:
        avg_loss:   average eval loss over all samples
        acc:    eval accuracy (%) over all samples in this epoch
    """
    model.eval()
    running_loss = 0.0
    correct = total = 0
    with torch.no_grad(): # disable gradient calculation
        for x, y in loader:
            x, y   = x.to(device), y.to(device)
            pred = model(x)
            loss   = criterion(pred, y)

            running_loss += loss.item() * x.size(0)
            preds = pred.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return running_loss / total, 100 * correct / total

#===================================================================================================================================
# MODIFY YOUR CODES HERE
"""
Add a momentum parameter to compare different momentum values
Call trainer.py's make_optimizer function
SEE DETAILS IN trainer.py
"""
from train.trainer import make_optimizer







#===================================================================================================================================

#===================================================================================================================================
# ADD YOUR CODES HERE
"""
Parameters modify: IMPORT hist, save_path from main
Call trainer.py's save_results function
SEE DETAILS IN trainer.py
"""
from train.trainer import save_results




#===================================================================================================================================

def main(ckpt, device, seed, lr, epochs, batch_size, optimizer_name="sgd", momentum=None):
    """
    Add a momentum parameter to compare different momentum values
    """
    # --- data ---
    train_loader, val_loader, test_loader = get_loaders(batch_size=batch_size)

    # --- seed: set random seeds for reproducibility ---
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # --- model / loss / optimizer ---
    model = VGG13(num_classes=10, input_channels=3, pretrained=False).to(device)
    criterion = nn.CrossEntropyLoss()

    ######## UPDATE#########
    # add a momentum parameter
    optimizer = make_optimizer(optimizer_name, model.parameters(), lr=lr, momentum=momentum)
    
    # --- save history for the plots ---
    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "epoch_time": []}
    
    # --------------- try to resume (epoch-level) ---------------
    start_epoch, hist = ckpt.resume(model, optimizer, hist)

    # --- train loop ---
    for epoch in range(start_epoch + 1, epochs + 1):
        t0 = time.perf_counter()
        
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = eval_one_epoch(model, val_loader, criterion, device)
        
        elapsed = time.perf_counter() - t0
        
        hist["train_loss"].append(tr_loss)
        hist["train_acc"].append(tr_acc)
        hist["val_loss"].append(va_loss)
        hist["val_acc"].append(va_acc)
        hist["epoch_time"].append(elapsed)

        print(f"[{optimizer_name}] Epoch {epoch}/{epochs} | "
          f"Train: loss={tr_loss:.4f}, acc={tr_acc:.2f}% | "
          f"Val: loss={va_loss:.4f}, acc={va_acc:.2f}% | "
          f"time: {elapsed:.2f}s")
        
        # ---- save one checkpoint per completed epoch (highly recommended) ----
        ckpt.save(epoch, model, optimizer, hist, config={"lr": lr, "batch_size": batch_size})

    # --- final test ---
    t1 = time.perf_counter()
    te_loss, te_acc = eval_one_epoch(model, test_loader, criterion, device)
    elapsed1 = time.perf_counter() - t1
    print(f"[{optimizer_name}] Test: loss={te_loss:.4f}, acc={te_acc:.2f}% | "
          f"time: {elapsed1:.2f}s")

    hist["test_loss"] = te_loss
    hist["test_acc"]  = te_acc
    ckpt.save(epochs, model, optimizer, hist, config={"lr": lr, "batch_size": batch_size})
    
    ######## UPDATE#########
    # save results
    summary_path = ckpt.path.replace('.pth', '_summary.txt') 
    save_results(hist, summary_path)

#===================================================================================================================================
# MODIFY YOUR CODES HERE
from train.trainer import run_experiment
if __name__ == "__main__":
    MODEL_TYPE_NAME = "3_cifar_vgg13"
    OUTPUT_DIR_MAIN = "3_cifar_vgg13_checkpoints"
    OUTPUT_DIR_MOMENTUM = "3_cifar_vgg13_checkpoints_momentum"

    run_experiment(
        main_func=main,
        model_type_name=MODEL_TYPE_NAME,
        output_dir_main=OUTPUT_DIR_MAIN,
        output_dir_momentum=OUTPUT_DIR_MOMENTUM
    )
    


