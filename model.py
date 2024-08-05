from sktime.classification.deep_learning import *



cl_dict = {
    "cnn": cnn(),
}

def cnn() :
    return CNNClassifier()