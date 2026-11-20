
import tensorflow as tf
from tensorflow.keras import layers, Model


class PositionalEncoding(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        dim = tf.shape(inputs)[2]

        positions = tf.cast(tf.range(seq_len)[:, None], tf.float32)
        dimensions = tf.cast(tf.range(dim)[None, :], tf.float32)

        angle_rates = 1.0 / tf.pow(
            10000.0,
            (2.0 * (dimensions // 2)) / tf.cast(dim, tf.float32)
        )

        angles = positions * angle_rates

        even_mask = tf.equal(tf.math.mod(dimensions, 2), 0)

        positional_encoding = tf.where(
            even_mask,
            tf.sin(angles),
            tf.cos(angles)
        )

        positional_encoding = positional_encoding[None, :, :]

        return inputs + positional_encoding


def transformer_encoder(x, d_model=64, num_heads=4, ff_dim=128,
                      dropout=0.2):

    attention = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads,
        dropout=dropout
    )(x, x)

    x = layers.Add()([x, attention])
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    ff = layers.Dense(ff_dim, activation="relu")(x)
    ff = layers.Dropout(dropout)(ff)
    ff = layers.Dense(d_model)(ff)

    x = layers.Add()([x, ff])
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    return x


def build_slm_transformer(input_features, num_classes,
                          d_model=64,
                          num_heads=4,
                          ff_dim=128,
                          dropout=0.3):

    inputs = layers.Input(
        shape=(1, input_features),
        name="input_features"
    )

    x = layers.Dense(d_model, activation="relu")(inputs)

    x = PositionalEncoding()(x)

    x = transformer_encoder(
        x,
        d_model=d_model,
        num_heads=num_heads,
        ff_dim=ff_dim,
        dropout=dropout
    )

    x = transformer_encoder(
        x,
        d_model=d_model,
        num_heads=num_heads,
        ff_dim=ff_dim,
        dropout=dropout
    )

    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="crop_prediction"
    )(x)

    model = Model(inputs, outputs, name="SLM_Transformer")

    return model


def compile_model(model, learning_rate=0.001):

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
