from sktime.classification.deep_learning import *
from sktime.classification.hybrid import *
from xgboost import XGBClassifier
from collections import Counter
import tensorflow as tf
import numpy as np


def get_model(model_name, input_shape=None, n_classes=None, y_train=None):
    cl_dict = {
        "gbm": XGBClassifier,
        # time series
        "lstm": LSTMFCNClassifier,
        # cnn
        "resnet": ResNetClassifier,
        # "inception": InceptionTimeClassifier,
        # transformer
        #"transformer" : MVTSTransformerClassifier,
    }
    print(model_name)
    if model_name in cl_dict:
        if model_name == "gbm":
            model = cl_dict[model_name](
                scale_pos_weight=Counter(y_train.ravel())[0] / Counter(y_train.ravel())[1],
                objective="binary:logistic",
                eval_metric=["auc", "logloss"],
                tree_method="gpu_hist",
                use_label_encoder=False,
                seed=0,
                n_estimators=1000,
                learning_rate=0.04,
                max_depth=5,
                min_child_weight=2,
                gamma=0.4,
                subsample=0.5,
                colsample_bytree=0.5,
            )
        elif model_name == "lstm" :
            # (n_, length, dimention)
            model = cl_dict[model_name]()
            model = model.build_model(input_shape=input_shape, n_classes=1)
            model.layers[-1].activation = tf.keras.activations.sigmoid

        elif  model_name == "resnet" :
            model = cl_dict[model_name](loss="binary_crossentropy")
            model = model.build_model(input_shape=input_shape, n_classes=1)

        elif  model_name == "inception" :
            model = cl_dict[model_name]()
            model = model.build_model(input_shape=input_shape, n_classes=1)
            model.layers()[-1].activation = tf.keras.activations.sigmoid

        # elif  model_name == "transformer" :
        #     model = HIVECOTEV2()



        #model = model.build_model(input_shape=input_shape, n_classes=1)
    elif model_name == "lstm_ori":
        inputs = []
        outputs = []
        concats = []

        
        input = tf.keras.layers.Input(shape=input_shape)
        inputs.append(input)
        concats.append(input)
        output = input


        llayer, lnode, fnode, droprate = 1, 64, 16, 0.5 # original paper hyperparameter

        if llayer:
            for _ in range(llayer - 1):
                output = tf.keras.layers.LSTM(lnode, return_sequences=True)(output)
            output = tf.keras.layers.LSTM(lnode)(output)
            concats.append(output)
        
        output = tf.keras.layers.Flatten()(output)#tf.keras.layers.concatenate(concats)
        
        output = tf.keras.layers.Dense(fnode, activation="relu")(output)
        output = tf.keras.layers.Dropout(droprate)(output)
        output = tf.keras.layers.Dense(
            1,
            activation="sigmoid",
            bias_initializer=tf.keras.initializers.Constant(np.log([Counter(y_train.ravel())[0] / Counter(y_train.ravel())[1]])),
        )(output)

        outputs.append(output)
        model = tf.keras.models.Model(inputs=inputs, outputs=[output])
    
    elif model_name == "transformer" :
        inputs = []
        outputs = []
        concats = []

        
        input = tf.keras.layers.Input(shape=input_shape)
        inputs.append(input)
        concats.append(input)
        output = input


        print("good")
        (nfilt, nhead, kdim, fnode, clayer, tlayer, 
                droprate, filtsize, poolsize,pooltype) = 64,3,32,32,1,3,0.2,5,4,"max"
        # conv
        for _ in range(clayer):
            output = tf.keras.layers.Conv1D(
                filters=nfilt,
                kernel_size=filtsize,
                padding="same",
                activation="relu",)(output)
            output = tf.keras.layers.MaxPooling1D(poolsize, padding="same")(output)
        # 마지막 차원이 nfilt. kdim으로 바꿔야 transformer block 쌓기 가능
        output = tf.keras.layers.Dense(kdim)(output)

        # transformer
        for _ in range(tlayer):
            attn_output = tf.keras.layers.MultiHeadAttention(
                num_heads=nhead,
                key_dim=kdim,
                attention_axes=[
                    1,
                ],
            )(output, output)
            attn_output = tf.keras.layers.Dropout(droprate)(attn_output)
            # sum and norm
            output1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(
                output + attn_output
            )

            ffn_output = tf.keras.layers.Dense(fnode, activation="relu")(output1)
            ffn_output = tf.keras.layers.Dense(kdim)(ffn_output)
            output2 = tf.keras.layers.Dropout(droprate)(ffn_output)
            # sum and norm
            output = tf.keras.layers.LayerNormalization(epsilon=1e-6)(
                output1 + output2
            )
        if pooltype == "avg":
            output = tf.keras.layers.GlobalAveragePooling1D()(output)
        else:
            output = tf.keras.layers.GlobalMaxPooling1D()(output)
        output = tf.keras.layers.Dropout(droprate)(output)
        concats.append(output)

        if len(concats) > 1:
            output = tf.keras.layers.Flatten()(output)
            #output = tf.keras.layers.concatenate(concats)
        output = tf.keras.layers.Dense(fnode, activation="relu")(output)
        if droprate:
            output = tf.keras.layers.Dropout(droprate)(output)
        output = tf.keras.layers.Dense(
            1,
            activation="sigmoid",
            bias_initializer=tf.keras.initializers.Constant(np.log([Counter(y_train.ravel())[0] / Counter(y_train.ravel())[1]])),
            )(output)

        outputs.append(output)
        print(inputs)
        print(output)
        model = tf.keras.models.Model(inputs=inputs, outputs=[output])
    


        
    else:
    
        raise ValueError(f"Model {model_name} is not recognized.")
    
    return model
