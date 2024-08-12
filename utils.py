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
)

import matplotlib.pyplot as plt
import numpy as np
import json


def visualization(history, paths):

    plt.subplot(3, 1, 1)
    plt.ylim([0, 1])
    plt.plot(history["loss"])
    plt.plot(history["val_loss"])
    plt.title("loss")

    last_accuracy = history["auc"][-1]
    last_val_accuracy = history["val_auc"][-1]

    plt.subplot(3, 1, 2)
    plt.plot(history["auc"])
    plt.plot(history["val_auc"])
    plt.text(len(history["auc"]) - 1, last_accuracy, f"{last_accuracy:.4f}", color="blue", ha="center", va="bottom")
    plt.text(len(history["val_auc"]) - 1, last_val_accuracy, f"{last_val_accuracy:.4f}", color="orange", ha="center", va="bottom")
    plt.title("auc")

    last_accuracy = history["prc"][-1]
    last_val_accuracy = history["val_prc"][-1]

    plt.subplot(3, 1, 3)
    plt.plot(history["prc"])
    plt.plot(history["val_prc"])
    plt.text(len(history["prc"]) - 1, last_accuracy, f"{last_accuracy:.4f}", color="blue", ha="center", va="bottom")
    plt.text(len(history["val_prc"]) - 1, last_val_accuracy, f"{last_val_accuracy:.4f}", color="orange", ha="center", va="bottom")
    plt.title("prc")

    plt.tight_layout()
    plt.savefig(paths)
    plt.clf()


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


def analyze(save_path, title, test_type, res):
    # ROC
    tprs = []
    aucs = []
    auprcs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots(figsize=(6, 6))
    for fold, (true, pred) in enumerate(res):
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
    plt.savefig(save_path / f"fig/{test_type}.png")

    # statistics

    print("AUROC(std) ", mean_auc, "(", std_auc, ")")
    auprcs = np.array(auprcs)
    print("AUPRC(std) ", auprcs.mean(), "(", auprcs.std(), ")")

    best_fold = np.array(aucs).argmax()
    true, pred = res[best_fold]

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

    print(thresholds[opt_idx])

    metrics = {
    "test type":test_type, 
    "theshold":thresholds[opt_idx],
    "Accuracy": np.round((true == (pred > thresholds[opt_idx])).mean(), 4),
    "Precision(PPV)": np.round(precision[opt_idx], 4),
    "NPV": np.round(NPV, 4),
    "Recall": np.round(recall[opt_idx], 4),
    "Specificity": np.round(calculate_specificity(pred, thresholds[opt_idx]), 4),
    "F1 score": np.round(max_f1_score, 4),
    "AUROC": np.round(aucs[best_fold], 4),
    "AUPRC": np.round(auprcs[best_fold], 4),
}

    # 성능 지표 출력
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

    # history.json 파일에 성능 지표 저장
    with open(save_path / "results.txt", "w") as f:
        json.dump(metrics, f, indent=4)



    np.save(save_path / f"npy/{test_type}_{title}_true.npy", np.array(true))
    np.save(save_path / f"npy/{test_type}_{title}_pred.npy", np.array(pred))

    viz = RocCurveDisplay.from_predictions(
        true,
        pred,
    )
    interp_tpr = np.interp(np.linspace(0, 1, 100), viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    np.save(save_path / f"npy/{test_type}_auroc_value.npy", np.array(interp_tpr))
    # plt.clf()
    precision, recall, _ = precision_recall_curve(true, pred)
    interp_recall = np.linspace(0, 1, 100)
    interp_precision = np.interp(interp_recall, recall[::-1], precision[::-1])
    np.save(save_path / f"npy/{test_type}_auprc_value.npy", np.array(interp_precision))


# import os
# import matplotlib.pyplot as plt

# def check_dataset() :

#     path = "./data/processed/CNUH/"
#     data_list = os.listdir(path)

#     for data_path in data_list :
#         data = np.load(path + data_path, allow_pickle=True)
#         print(data.shape)

#         fig = plt.figure(figsize=(14,12))
#         for i in range(data.shape[1]) :
#             plt.subplot(data.shape[1], 1, i+1)
#             plt.plot(data[:, i].reshape(-1,))
#         plt.savefig("all_variable_visualization.png")
#         break


# if __name__ == "__main__" :
#     print(check_dataset())
