from sklearn.metrics import (
    accuracy_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
    RocCurveDisplay,
    ConfusionMatrixDisplay,
    fbeta_score
)

import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import io
import tensorflow as tf
from keras import backend as K
from keras.metrics import Metric
from tensorboard.plugins.hparams import api as hp
from pathlib import Path


class FBetaScore(Metric):
    def __init__(self, beta=1, name="fbeta_score", **kwargs):
        super(FBetaScore, self).__init__(name=name, **kwargs)
        self.beta = beta
        self.true_positives = self.add_weight(name="tp", initializer="zeros")
        self.false_positives = self.add_weight(name="fp", initializer="zeros")
        self.false_negatives = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.round(y_pred)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum((1 - y_true) * y_pred)
        fn = tf.reduce_sum(y_true * (1 - y_pred))

        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

    def result(self):
        beta_square = self.beta ** 2
        tp = self.true_positives
        fp = self.false_positives
        fn = self.false_negatives

        precision = tp / (tp + fp + K.epsilon())
        recall = tp / (tp + fn + K.epsilon())

        fbeta_score = (1 + beta_square) * precision * recall / (beta_square * precision + recall + K.epsilon())
        return fbeta_score

    def reset_state(self):
        self.true_positives.assign(0)
        self.false_positives.assign(0)
        self.false_negatives.assign(0)


def plot_roc_curve(res, title, save_path):
    # Prepare the ROC curve plot
    fig, ax = plt.subplots(figsize=(6, 6))
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []
    auprcs = []

    for fold, (true, pred, y_all) in enumerate(res):
        viz = RocCurveDisplay.from_predictions(
            true,
            pred,
            name=f"ROC fold {fold+1}",
            alpha=0.3,
            lw=1,
            ax=ax,
            plot_chance_level=(fold == 4),
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

        # AUPRC
        precision, recall, thresholds = precision_recall_curve(true, pred)
        auprcs.append(auc(recall, precision))

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
    )
    ax.legend(loc="lower right")

    plt.savefig(str(save_path / "roc.png"))

    return tprs, aucs, auprcs


def analyze(log_dir, title, test_type, res, params):
    # ROC
    save_path = Path(log_dir) / test_type
    os.makedirs(str(save_path), exist_ok=True)

    _, aucs, auprcs = plot_roc_curve(res, title, save_path)

    aucs = np.array(aucs)
    auprcs = np.array(auprcs)
    print(test_type)
    print("AUROC(std) ", aucs.mean(), "(", aucs.std(), ")")
    print("AUPRC(std) ", auprcs.mean(), "(", auprcs.std(), ")")

    best_fold = aucs.argmax()
    true, pred, y_all = res[best_fold]

    precision, recall, thresholds = precision_recall_curve(true, pred)

    numerator = 2 * recall * precision
    denom = recall + precision  
    f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom != 0))
    
    beta = 2
    numerator = (1+(beta**2)) * recall * precision
    denom = recall + (beta**2) * precision
    f2_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom != 0))

    opt_idx = np.where(f2_scores == np.max(f2_scores))[0][0]
    th = thresholds[opt_idx]
    y_pred_opt = (pred > th).astype(int)

    print("Threshold :", th)
    metrics = {
        "threshold": float(th),
        "Accuracy": float(np.round(accuracy_score(true, y_pred_opt), 4)),
        "Precision": float(np.round(precision[opt_idx], 4)),
        "Recall(PPV)": float(np.round(recall[opt_idx], 4)),
        "F1 score": float(np.round(f1_scores[opt_idx], 4)),
        "F2 score" : float(np.round(f2_scores[opt_idx], 4)),
        "AUROC": float(np.round(aucs[best_fold], 4)),
        "AUPRC": float(np.round(auprcs[best_fold], 4)),
        
    }

    hparams_dict = {"test_type" : params[0],
                   "learning_win" : params[1],
                   "pred_win" : params[2],
                   "sampling_rate" : params[3],
                   "batch_size" : params[4],
                   "learning_rate" : params[5],
                   }
    
    
    with tf.summary.create_file_writer(str(save_path)).as_default():
        hp.hparams(hparams_dict, trial_id=save_path.parent.name)

        tf.summary.scalar("auc", metrics["AUROC"], step=1)
        tf.summary.scalar("prc", metrics["AUPRC"], step=1)
        tf.summary.scalar("f1score", metrics["F1 score"], step=1)
        tf.summary.scalar("f2score", metrics["F2 score"], step=1)
        tf.summary.scalar("threshold", metrics["threshold"], step=1)
        
        tf.summary.scalar("Accuracy", metrics["Accuracy"], step=1)
        tf.summary.scalar("Precision", metrics["Precision"], step=1)
        tf.summary.scalar("Recall(PPV)", metrics["Recall(PPV)"], step=1)


    ConfusionMatrixDisplay.from_predictions(
        true,
        y_pred_opt,
        cmap=plt.cm.Blues
    )
    plt.savefig(str(save_path / "CM_all.png"))
    mask = y_all == 1
    ConfusionMatrixDisplay.from_predictions(
        true[mask],
        y_pred_opt[mask],
        cmap=plt.cm.Blues
    )
    plt.savefig(str(save_path / "CM_hypoxemia.png"))
