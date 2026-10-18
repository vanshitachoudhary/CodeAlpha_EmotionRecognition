"""Deep models (Keras): 1D-CNN, BiLSTM, CNN-BiLSTM on (188 frames x 80 MFCC+delta) sequences.
TensorFlow optional hai: install na ho to train.py classical models ke saath chalta hai."""
import importlib.util

NAMES = ["cnn", "bilstm", "cnn_lstm"]
TITLES = {"cnn": "1D-CNN", "bilstm": "BiLSTM", "cnn_lstm": "CNN-BiLSTM"}


def available():
    return importlib.util.find_spec("tensorflow") is not None


def build(name, input_shape, n_classes, seed=42):
    from tensorflow import keras
    from tensorflow.keras import layers
    keras.utils.set_random_seed(seed)

    def conv_block(x, filters, drop=0.25):
        x = layers.Conv1D(filters, 5, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling1D(2)(x)
        return layers.Dropout(drop)(x)

    pool = lambda x: layers.Concatenate()([layers.GlobalAveragePooling1D()(x), layers.GlobalMaxPooling1D()(x)])
    inp = keras.Input(shape=input_shape)
    x = inp
    if name == "cnn":
        for f in (64, 128, 192):
            x = conv_block(x, f)
    elif name == "bilstm":
        x = layers.AveragePooling1D(2)(x)                      # 188 -> 94 steps: CPU pe LSTM tez
        x = layers.Bidirectional(layers.LSTM(96, return_sequences=True))(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    elif name == "cnn_lstm":
        for f in (64, 128):
            x = conv_block(x, f)
        x = layers.Bidirectional(layers.LSTM(96, return_sequences=True))(x)
    else:
        raise ValueError(f"unknown model {name}")
    x = pool(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inp, out, name=name)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def fit(model, Xtr, ytr, Xva, yva, epochs=60, batch=32, class_weight=None, verbose=2):
    from tensorflow import keras
    cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
          keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5)]
    h = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs, batch_size=batch,
                  class_weight=class_weight, callbacks=cb, verbose=verbose)
    return {k: [float(v) for v in vals] for k, vals in h.history.items()}
