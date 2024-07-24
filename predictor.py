import sys
sys.path.append("modules")
import numpy as np
import pandas as pd
import os
import math
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.models as models
from utils import _get_image_labels
from dataset import SmokeDataset
from Configs import Config
from sklearn.metrics import confusion_matrix, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')
cudnn.benchmark = True

folds = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5']

def predict(test_loader, model, cfg):
    model.eval()
    y_pred = []
    y_true = []
    y_prob = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            images = inputs.to(cfg.device)
            target = labels.to(cfg.device)
            output = model(images)
            output = torch.sigmoid(output).cpu()
            output = output.detach().numpy()
            y_prob.extend(output)
            output = (output>0.5).astype(np.float32)
            target = target.cpu()
            y_true.extend(target)
            y_pred.extend(output)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(confusion_matrix(y_true, y_pred))
    mcc = float(tp*tn - fp*fn) / (math.sqrt((tp+fp) * (tp+fn) * (tn+fp) * (tn+fn)) + 1e-6)
    sn = float(tp) / (tp+fn +1e-6)
    sp = float(tn) / (tn+fp +1e-6)
    acc = float(tp+tn) / (tn+fp+fn+tp +1e-6)
    fpr, tpr, _ = roc_curve(y_true, y_prob, pos_label=1)
    auc_ = auc(fpr, tpr)
    pre = float(tp) / (tp+fp + 1e-6)
    F1_ = 2 * (pre * sn) / (pre + sn)
    return mcc, sn, sp, acc, auc_, pre, F1_

def run():
    cfg = Config()
    
    print('Creating testing set')

    test_images, test_labels = _get_image_labels(cfg.dataset_dir)

    test_dataset = SmokeDataset(
        test_images, test_labels, resize=cfg.resize, transform=cfg.test_transform)


    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=True, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
    )
    
    mcc_test_scores = []
    sn_test_scores = []
    sp_test_scores = []
    acc_test_scores = []
    auc_test_scores = []
    pre_test_scores = []
    f1_test_scores = []

    for fold in folds:
        print(fold)
        
        model = getattr(models, cfg.model_name)(
            pretrained=False, num_classes=cfg.num_classes,)

        if os.path.isfile(cfg.model_path + "{}".format(fold) + ".pth"):
            model.load_state_dict(torch.load(cfg.model_path + "{}".format(fold) + ".pth", map_location=cfg.device))
            print('Saved model found at', cfg.model_path ,' and loaded')

        model = model.to(cfg.device)
    
        mcc, sn, sp, acc, auc_, pre, F1_ = predict(test_loader, model, cfg)

        mcc_test_scores.append(mcc)
        sn_test_scores.append(sn)
        sp_test_scores.append(sp)
        acc_test_scores.append(acc)
        auc_test_scores.append(auc_)
        pre_test_scores.append(pre)
        f1_test_scores.append(F1_)
     
    print("Independent Testing Scores {}\t{}\t{}\t{}\t{}\t{}\t{}".format(np.round(np.mean(mcc_test_scores), 4), np.round(np.mean(sn_test_scores), 4), np.round(np.mean(sp_test_scores), 4), np.round(np.mean(acc_test_scores), 4), np.round(np.mean(auc_test_scores), 4), np.round(np.mean(pre_test_scores), 4), np.round(np.mean(f1_test_scores), 4)))
    
if __name__ == '__main__':
    run()

