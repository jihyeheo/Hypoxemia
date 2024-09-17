from sktime.classification.deep_learning import LSTMFCNClassifier, ResNetClassifier, InceptionTimeClassifier
import tensorflow as tf
from tensorflow import keras
from keras import layers, Model

# Define the hidden layer as a custom Keras layer
class HiddenLayer(layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.linear = layers.Dense(units)
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()

    def call(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ConvLayer(layers.Layer):
    def __init__(self, filters):
        super().__init__()
        self.linear = layers.Conv1D(filters=filters, kernel_size=3, strides=1)
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.pool = layers.MaxPool1D()

    def call(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

class MetaModel(layers.Layer):
    def __init__(self):
        super().__init__()
        self.layer1 = HiddenLayer(32)
        self.layer2 = HiddenLayer(32)

    def call(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class SingleModel(layers.Layer):
    def __init__(self):
        super().__init__()

        self.layer1 = ConvLayer(32)
        self.layer2 = ConvLayer(32)
        self.layer3 = ConvLayer(1)
        self.flatten = layers.Flatten()
        self.layer4 = HiddenLayer(32)

    def call(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.flatten(x)
        x = self.layer4(x)
        return x

# Define the main model
class WaveSingleMetaModel(Model):
    def __init__(self, backbone, **kwargs):
        super().__init__(**kwargs)

        self.wave = backbone
        self.single = SingleModel()
        self.meta = MetaModel()
        self.layer1 = HiddenLayer(32)
        self.layer2 = HiddenLayer(32)
        self.fc = layers.Dense(1)
        self.sig = layers.Activation('sigmoid')


    def call(self, input):
        wave, single, meta = input 
        x = tf.concat([self.wave(wave), self.single(single), self.meta(meta)], axis=1) # B x features
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.fc(x)
        x = self.sig(x)
        return x

class WaveSingleModel(Model):
    def __init__(self, backbone):
        super().__init__()

        self.wave = backbone
        self.single = SingleModel()
        self.layer1 = HiddenLayer(32)
        self.layer2 = HiddenLayer(32)
        self.fc = layers.Dense(1)
        self.sig = layers.Activation('sigmoid')


    def call(self, input):
        wave, single, meta = input 
        x = tf.concat([self.wave(wave), self.single(single)], axis=1) # B x features
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.fc(x)
        x = self.sig(x)
        return x
    
class WaveMetaModel(Model):
    def __init__(self, backbone):
        super().__init__()

        self.wave = backbone
        self.meta = MetaModel()
        self.layer1 = HiddenLayer(32)
        self.layer2 = HiddenLayer(32)
        self.fc = layers.Dense(1)
        self.sig = layers.Activation('sigmoid')


    def call(self, input):
        wave, single, meta = input
        x = tf.concat([self.wave(wave), self.meta(meta)], axis=1) # B x features
        x = self.meta(meta)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.fc(x)
        x = self.sig(x)
        return x

class WaveModel(Model):
    def __init__(self, backbone):
        super().__init__()

        self.wave = backbone
        self.layer1 = HiddenLayer(32)
        self.layer2 = HiddenLayer(32)
        self.fc = layers.Dense(1)
        self.sig = layers.Activation('sigmoid')

    def call(self, input):
        wave, single, meta = input 
        x = self.wave(wave) # B x features
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.fc(x)
        x = self.sig(x)
        return x

def get_model(backbone, input_shape, use_single=True, use_meta=True, feature_len=32):

    cl_dict = {
        "lstm": LSTMFCNClassifier,
        "resnet": ResNetClassifier,
        "inception": InceptionTimeClassifier,
    }

    wave_model = cl_dict[backbone]()
    wave_model = wave_model.build_model(input_shape=input_shape, n_classes=feature_len)
    wave_model.layers[-1].activation = tf.keras.activations.relu
    wave_model.save(f"{backbone}.keras")
    backbone = keras.models.load_model(f"{backbone}.keras", compile=False)

    if use_single and use_meta:
        return WaveSingleMetaModel(backbone)
    elif use_single:
        return WaveSingleModel(backbone)
    elif use_meta:
        return WaveMetaModel(backbone)
    else:
        return WaveModel(backbone)
