from sktime.classification.deep_learning import *
from xgboost import XGBClassifier
from collections import Counter

def get_model(model_name, input_shape=None, n_classes=None, y_train=None):
    cl_dict = {
        'gbm' : "asds",
        'cnn': CNNClassifier,
        'cntc': CNTCClassifier,
        'fcn': FCNClassifier,
        'lstm': LSTMFCNClassifier,
        'resnet': ResNetClassifier
    }

    if model_name in cl_dict:
        if model_name == "gbm" :
            model = XGBClassifier(scale_pos_weight=Counter(y_train.ravel())[0]/Counter(y_train.ravel())[1],
                                    objective="binary:logistic",
                                    eval_metric=["auc", "logloss"],
                                    tree_method="gpu_hist",)
        else : 

            model = cl_dict[model_name]()
            model = model.build_model(input_shape=input_shape, n_classes=1)
    else:
        raise ValueError(f"Model {model_name} is not recognized.")
    
    return model 