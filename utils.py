from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
    RocCurveDisplay,
    ConfusionMatrixDisplay,
    fbeta_score
)

import matplotlib.pyplot as plt
import numpy as np
import json
import os
from PIL import Image
import io

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import Metric
from tensorboard.plugins.hparams import api as hp

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


def calculate_specificity(true_pred, thresholds):
    # True Negative (TN)과 False Positive (FP) 초기화
    TN = 0
    FP = 0

    # 임계값(threshold)마다 TN과 FP를 계산
    for i in range(len(true_pred)):
        if true_pred[i] < thresholds:
            TN += 1
        else:
            FP += 1

    # Specificity 계산
    specificity = TN / (TN + FP)

    return specificity

def pretty_json(hp):
  json_hp = json.dumps(hp, indent=2)
  return "".join("\t" + line for line in json_hp.splitlines(True))


def plot_confusion_matrix(true_labels, predictions):
    # Create a confusion matrix display
    disp = ConfusionMatrixDisplay.from_predictions(
        true_labels,
        predictions,
        cmap=plt.cm.Blues
    )
    
    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    
    # Read the BytesIO object into a PIL Image
    image = Image.open(buf)
    
    # Convert the PIL Image to a NumPy array
    image_np = np.array(image)
    
    # Normalize the image if necessary (TensorBoard expects [0, 1] range)
    if image_np.max() > 1:
        image_np = image_np / 255.0
    
    return image_np

def plot_roc_curve(res, title):
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
        title=title,
    )
    ax.legend(loc="lower right")

    # Save plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    # Convert BytesIO object to a NumPy array
    image = Image.open(buf)
    image_np = np.array(image)

    # Normalize image if necessary
    if image_np.max() > 1:
        image_np = image_np / 255.0

    return tprs, aucs, auprcs, image_np


def analyze(save_path, title, test_type, res, windows):
    # ROC
    # 디렉토리 경로를 문자열로 생성
    npy_dir = os.path.join(save_path, f"lw{windows[0]},pw{windows[1]}", "npy")

    # 디렉토리가 없으면 생성
    for directory in [npy_dir]:
        if not os.path.isdir(directory):
            os.makedirs(directory)
    
    learning_win = windows[0]
    pred_win = windows[1]
    log_dir = save_path + f"/lw{learning_win},pw{pred_win}/{test_type}/"

    _, aucs, auprcs, image = plot_roc_curve(res, title)
    with tf.summary.create_file_writer(log_dir).as_default():
        tf.summary.image(f"{test_type} ROC Curve", image[None, ...], step=0)

    aucs = np.array(aucs)
    auprcs = np.array(auprcs)
    print("AUROC(std) ", aucs.mean(), "(", aucs.std(), ")")
    print("AUPRC(std) ", auprcs.mean(), "(", auprcs.std(), ")")

    best_fold = np.array(aucs).argmax()
    true, pred, y_all = res[best_fold]

    precision, recall, thresholds = precision_recall_curve(true, pred)
    numerator = 2 * recall * precision
    denom = recall + precision
    f1_scores = []

    f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom != 0))
    max_f1_score = np.max(f1_scores)
    opt_idx = np.where(f1_scores == max_f1_score)[0][0]
    y_pred_opt = (pred > thresholds[opt_idx]).astype(int)
    tn, fp, fn, tp = confusion_matrix(true, y_pred_opt).ravel()
    NPV = tn / (tn + fn) if (tn + fn) > 0 else 0
    f2_score = fbeta_score(true, y_pred_opt, beta=2)

    metrics = {
        "test type": test_type,
        "learning window" : learning_win,
        "pred window" : pred_win,
        "theshold": float(thresholds[opt_idx]),
        "Accuracy": float(np.round(accuracy_score(true, y_pred_opt), 4)),
        "Precision": float(np.round(precision[opt_idx], 4)),
        "NPV": float(np.round(NPV, 4)),
        "Recall(PPV)": float(np.round(recall[opt_idx], 4)),
        "Specificity": float(np.round(calculate_specificity(pred, thresholds[opt_idx]), 4)),
        "F1 score": float(np.round(max_f1_score, 4)),
        "F2 score" : float(np.round(f2_score, 4)),
        "AUROC": float(np.round(aucs[best_fold], 4)),
        "AUPRC": float(np.round(auprcs[best_fold], 4)),
        
    }


    with tf.summary.create_file_writer(log_dir).as_default():
        hp.hparams({"test_type":metrics["test type"]})
        tf.summary.scalar("Specificity", metrics["Specificity"], step=1)
        tf.summary.scalar("auc", metrics["AUROC"], step=1)
        tf.summary.scalar("prc", metrics["AUPRC"], step=1)
        tf.summary.scalar("f1score", metrics["F1 score"], step=1)
        tf.summary.scalar("f2score", metrics["F2 score"], step=1)
        tf.summary.scalar("Accuracy", metrics["Accuracy"], step=1)
        tf.summary.scalar("Precision", metrics["Precision"], step=1)
        tf.summary.scalar("Recall(PPV)", metrics["Recall(PPV)"], step=1)
        tf.summary.scalar("NPV", metrics["NPV"], step=1)



    # history.json 파일에 성능 지표 저장
    #file_path = save_path + "/results.json"

    # with open(file_path, "a") as f:
    #     json.dump(metrics, f, indent=4)

    np.save(npy_dir + f"/{test_type}_{title}_true.npy", np.array(true))
    np.save(npy_dir + f"/{test_type}_{title}_pred.npy", np.array(pred))

    viz = RocCurveDisplay.from_predictions(
        true,
        pred,
    )
    interp_tpr = np.interp(np.linspace(0, 1, 100), viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    np.save(npy_dir + f"/{test_type}_auroc_value.npy", np.array(interp_tpr))
    precision, recall, _ = precision_recall_curve(true, pred)
    interp_recall = np.linspace(0, 1, 100)
    interp_precision = np.interp(interp_recall, recall[::-1], precision[::-1])
    np.save(npy_dir + f"/{test_type}_auprc_value.npy", np.array(interp_precision))

    
    disp = plot_confusion_matrix(true, y_pred_opt)
    
    mask = y_all == 1
    disp2 = plot_confusion_matrix(true[mask], y_pred_opt[mask])

    with tf.summary.create_file_writer(log_dir).as_default():
        tf.summary.image("confusion matrix all", disp[None, ...], step=0)
        tf.summary.image("confusion matrix hypoxemia", disp2[None, ...], step=0)

