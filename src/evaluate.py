# Core PyTorch
import torch

# Scikit-learn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

# Utils
import src.utils as utils
import random
from random import sample  
import numpy as np
import copy
from tqdm import tqdm


def spatial_evaluate(model, model_name, loader, verbose = True, 
                     plt_confusion_matrix = True, show_plot = False, save_plot_path = None, device='cpu'):
    model = model.to(device)
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for data in loader:
              data = data.to(device)
              out, _, _ = model(data)
              preds.append(out.argmax(dim=1).detach().cpu().numpy())   # shape: [B]
              targets.append(data.state.detach().cpu().numpy())        # shape: [B]

    targets = np.concatenate(targets)
    preds = np.concatenate(preds)
    targets = [int(target) for target in targets]
    preds = [int(pred) for pred in preds]

    report_dict = classification_report(targets, preds, digits=4, output_dict=True, zero_division=0)
    if verbose == True:
      report = classification_report(targets, preds, digits=4, zero_division=0)
      print(f'Model: {model_name}')
      print('====================')
      print(report)

    if plt_confusion_matrix == True:
      utils.generate_cm(targets, preds, model_name, show=show_plot, save_path=save_plot_path)

    return report_dict
