import torch

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

from sklearn.preprocessing import (label_binarize)

import src.utils as utils
import numpy as np
import copy
from tqdm import tqdm
import random



def supcon_by_label(x, y, tau: float = 0.07, eps: float = 1e-12):
    """
    x : [B, D]  or [B, N, D]  (node dimension will be mean-pooled)
    y : [B] int (class labels, e.g., 0/1)
    tau : temperature
    """
    # 1) mean-pool if node dimension present
    if x.dim() == 3:
        x = x.mean(dim=1)          # [B, D]

    # 2) safe L2-normalize
    x = x / x.norm(dim=-1, keepdim=True).clamp_min(eps)

    B = x.size(0)
    if B < 2:
        return x.new_zeros(()).requires_grad_()

    # 3) similarity logits (fp32 for stability)
    x = x.float()
    logits = (x @ x.t()) / max(1e-3, min(1e2, float(tau)))   # [B, B]

    # 4) mask self with large negative (avoid -inf in logsumexp)
    eye = torch.eye(B, dtype=torch.bool, device=x.device)
    big_neg = torch.finfo(logits.dtype).min / 2
    logits = logits.masked_fill(eye, big_neg)

    # 5) supervised positives: same label, not self
    y = y.view(-1, 1)
    pos = (y == y.t()) & (~eye)                               # [B, B]
    pos_cnt = pos.sum(dim=1)                                  # [B]
    valid = pos_cnt > 0
    if not valid.any():
        return x.new_zeros(()).requires_grad_()

    # 6) row-wise log-softmax and mean over positives
    log_den = torch.logsumexp(logits, dim=1, keepdim=True)
    logp = logits - log_den                                   # [B, B]
    loss_i = torch.zeros(B, device=x.device, dtype=x.dtype)
    loss_i[valid] = - (logp[valid] * pos[valid]).sum(dim=1) / pos_cnt[valid].float().clamp_min(1)

    # 7) optional scale normalization to reduce batch-size effects
    loss = loss_i.mean() / torch.log(torch.tensor(B-1.0, device=x.device))

    # 8) last-ditch guard
    if not torch.isfinite(loss):
        return (x.sum() * 0.0)

    return loss




def spatial_train(model, train_loader, val_loader, num_epoch, patience, 
                  lr=0.0005, weight_decay=5e-4, smoothing=0, con_weight = 0.5, verbose = True, device = 'cpu'):
  model = model.float().to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
  loss_function = torch.nn.CrossEntropyLoss(label_smoothing=smoothing)
  #loss_function = torch.nn.BCEWithLogitsLoss()

  logs = {'train':{}, 'val':{}}
  logs['train'] = {
     'loss': [],
     'acc': [],
     'f1': [],
     'rec': [],
     'prec': [],
     'classf1': [],
     'auc': []
  }
  logs['val'] = {
     'loss': [],
     'acc': [],
     'f1': [],
     'rec': [],
     'prec': [],
     'classf1': [],
     'auc': []
  }

  best_train_rec = None
  best_val_rec = None
  best_model = None
  val_struggle = 0
  best_epoch = None
  struggle = 0

  weights = torch.tensor(np.ones(4)).float()  # Avoid divide by zero
  weights = weights / weights.sum()  # Normalize

  for epoch in tqdm(range(num_epoch)):
      model.train()
      probs, preds, targets = [], [], []
      total_loss = 0

      for data in train_loader:
          optimizer.zero_grad()  

          data = data.to(device)
            
          out, embs, edge_idxs = model(data)  # out shape: [8, 4]
          graph_embs = embs[1]
    
          prob = out.detach().cpu().numpy()  # shape: [B, 4]
          preds.append(out.argmax(dim=1).detach().cpu().numpy())   # shape: [B]
          targets.append(data.state.detach().cpu().numpy())        # shape: [B]
          y = torch.tensor(data.state).to(device).long()
          probs.append(prob)                                       # shape: [B, 4]
          target = torch.zeros(4)
          target[data.state] = 1
          task_loss = loss_function(out, data.state.to(device))         # data.state shape: [B]
          con_loss = supcon_by_label(graph_embs, y)
          loss = (1-con_weight) * task_loss + con_weight * con_loss
          loss.backward()
          optimizer.step()
          total_loss += loss.item()

      # Flatten to 1D arrays
      targets = np.concatenate(targets)
      preds = np.concatenate(preds)
      train_acc = accuracy_score(targets, preds)
      train_f1 = f1_score(targets, preds, average='macro')
      train_rec = recall_score(targets, preds, average='macro', zero_division=np.nan)
      train_classrec = recall_score(targets, preds, average=None)
      f_train_classrec = [f'{rec:.5f}' for rec in train_classrec]
      train_classrec_std = train_classrec.std()
      penalized_rec = train_rec * (1-train_classrec_std)
      train_prec = precision_score(targets, preds, average='macro', zero_division=np.nan)
      total_loss = total_loss/len(train_loader)
      train_classf1 = f1_score(targets, preds, average=None).tolist()
      f_train_classf1 = [f'{f1:.5f}' for f1 in train_classf1]
      classes = np.unique(targets)
      targets_bin = label_binarize(np.array(targets), classes=classes)
      probs = np.vstack(probs)
      try:
          auc = roc_auc_score(targets_bin, probs, average='macro', multi_class='ovo')
          class_auc = roc_auc_score(targets_bin, probs, average=None, multi_class='ovo')  
          std_auc = class_auc.std()
          auc = auc *(1-std_auc)
      except:
          auc = np.nan  # handle cases where AUC fails (e.g., only one class present)
          std_auc = np.nan

      logs['train']['loss'].append(total_loss)
      logs['train']['acc'].append(train_acc)
      logs['train']['f1'].append(train_f1)
      logs['train']['rec'].append(train_rec)
      logs['train']['prec'].append(train_prec)
      logs['train']['classf1'].append(train_classf1)
      logs['train']['auc'].append(auc)

      if best_train_rec == None:
        best_train_rec = penalized_rec
        best_model = copy.deepcopy(model)
        best_epoch = epoch + 1

      if val_loader == None:
        if penalized_rec > best_train_rec:
          best_model = copy.deepcopy(model)
          best_epoch = epoch + 1
          struggle = 0
        else: 
           struggle += 1
           if struggle > patience:
              return model, best_model, best_epoch, logs

      best_train_rec = max(penalized_rec, best_train_rec)

      if verbose:
          print(f"\n[Epoch {epoch+1}] \nTrain Loss: {total_loss:.5f}, Train Acc: {train_acc:.5f}"
                f"\nF1: {train_f1:.5f}, Class F1: {f_train_classf1}"
                f"\nPrec: {train_prec:.5f}, Rec: {penalized_rec:.5f}, Class Rec: {f_train_classrec}"
                f"\nTrain STD penalized AUC: {auc:.5f}, Train AUC STD: {std_auc:.5f}")

      if not val_loader == None:
        model.eval()
        val_probs, val_preds, val_targets = [], [], []
        val_loss = 0

        with torch.no_grad():
            for val_data in val_loader:
                val_data = val_data.to(device)
                out, embs, edge_idxs = model(val_data)

                val_prob = torch.softmax(out, dim=1).detach().cpu().numpy()
                val_probs.append(val_prob)

                val_preds.append(out.argmax(dim=1).detach().cpu().numpy())
                val_targets.append(val_data.state.detach().cpu().numpy())

                val_loss += loss_function(out, val_data.state.to(device)).item()

        # Flatten to 1D arrays
        val_targets = np.concatenate(val_targets)
        val_preds = np.concatenate(val_preds)
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average='macro')
        val_rec = recall_score(val_targets, val_preds, average='macro', zero_division=np.nan)
        val_classrec = recall_score(val_targets, val_preds, average=None)
        f_val_classrec = [f'{rec:.5f}' for rec in val_classrec]
        val_classrec_std = val_classrec.std()
        penalized_val_rec = val_rec * (1-val_classrec_std)
        val_prec = precision_score(val_targets, val_preds, average='macro', zero_division=np.nan)
        val_loss = val_loss/len(val_loader)
        val_classf1 = f1_score(val_targets, val_preds, average=None).tolist()
        f_val_classf1 = [f'{f1:.5f}' for f1 in val_classf1]
        classes = np.unique(targets)
        val_targets_bin = label_binarize(np.array(targets), classes=classes)
        try:
            val_auc = roc_auc_score(val_targets_bin, val_probs, average='macro', multi_class='ovo')
            val_class_auc = roc_auc_score(val_targets_bin, val_probs, average=None, multi_class='ovo')  
            val_std_auc = val_class_auc.std()
            val_auc = val_auc*(1-val_std_auc)
        except:
            val_auc = np.nan  # handle cases where AUC fails (e.g., only one class present)
            val_std_auc = np.nan

        logs['val']['loss'].append(val_loss)
        logs['val']['acc'].append(val_acc)
        logs['val']['f1'].append(val_f1)
        logs['val']['rec'].append(val_rec)
        logs['val']['prec'].append(val_prec)
        logs['val']['classf1'].append(val_classf1)
        logs['val']['auc'].append(val_auc)

        if verbose:
            print(f"\nVal Loss: {val_loss:.5f}, Val Acc: {val_acc:.5f}"
                  f"\nF1: {val_f1:.5f}, Class F1: {f_val_classf1}"
                  f"\nPrec: {val_prec:.5f}, Rec: {penalized_val_rec:.5f}, Class Rec: {f_val_classrec}"
                  f"\nVal STD penalized AUC: {val_auc:.5f}, ")
      
        if best_val_rec == None:
          best_val_rec = penalized_val_rec
          best_model = copy.deepcopy(model)
          best_epoch = epoch + 1
        if penalized_val_rec > best_val_rec:
          val_struggle = 0
          best_model = copy.deepcopy(model)
          best_epoch = epoch + 1
        else: 
          val_struggle += 1
          if val_struggle > patience:
             return model, best_model, best_epoch, logs
        best_val_rec = max(penalized_val_rec, best_val_rec)

  return model, best_model, best_epoch, logs # final model, best_model



