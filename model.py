from sktime.classification.deep_learning import *
from xgboost import XGBClassifier
from collections import Counter


def get_model(model_name, input_shape=None, n_classes=None, y_train=None):
    cl_dict = {
        "gbm": XGBClassifier,
        "cnn": CNNClassifier,
        "cntc": CNTCClassifier,
        "fcn": FCNClassifier,
        "lstm": LSTMFCNClassifier,
        "resnet": ResNetClassifier,
        "inception": InceptionTimeClassifier,
    }

    if model_name in cl_dict:
        if model_name == "gbm":
            model = cl_dict[model_name](
                scale_pos_weight=Counter(y_train.ravel())[0] / Counter(y_train.ravel())[1],
                objective="binary:logistic",
                eval_metric=["auc", "logloss"],
                tree_method="gpu_hist",
                use_label_encoder=False,
                seed=0,
                n_estimators=2000,
                learning_rate=0.04,
                max_depth=5,
                min_child_weight=2,
                gamma=0.4,
                subsample=0.5,
                colsample_bytree=0.5,
            )
        else:
            if model_name not in ["inception", "cntc", "lstm"] :
                model = cl_dict[model_name](activation="sigmoid",loss="binary_crossentropy")
            elif  model_name == "lstm" :
                model = cl_dict[model_name]()
            else :
                model = cl_dict[model_name](loss="binary_crossentropy")
            model = model.build_model(input_shape=input_shape, n_classes=1)
    else:
        raise ValueError(f"Model {model_name} is not recognized.")

    return model
