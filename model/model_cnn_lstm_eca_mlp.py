import math
from keras.api import layers, Model
from keras.api.layers import Input, Conv1D, LSTM, Flatten, Dense, Dropout, BatchNormalization, Activation, MaxPooling1D
from keras.api.regularizers import l2

class ECABlock1D(layers.Layer):

    def __init__(self, b=1, gamma=2):
        super().__init__()
        self.b = b
        self.gamma = gamma

    def build(self, input_shape):
        channels = input_shape[-1]

        kernel_size = int(abs((math.log(channels, 2) + self.b) / self.gamma))
        if kernel_size % 2 == 0:
            kernel_size += 1

        self.conv = Conv1D(
            filters=1,
            kernel_size=kernel_size,
            padding='same',
            use_bias=False,
            kernel_initializer='he_normal'
        )

        super().build(input_shape)

    def call(self, inputs):

        x = layers.GlobalAveragePooling1D()(inputs)
        x = layers.Reshape((-1, 1))(x)

        x = self.conv(x)
        x = Activation('sigmoid')(x)

        x = layers.Reshape((1, -1))(x)
        return layers.multiply([inputs, x])


def build_spectrum_model(
        input_shape,
        conv_filters=[256, 64],
        kernel_sizes=[5, 2],
        use_eca=True,
        lstm_units=700,
        dense_units=[256, 256, 64],
        use_dense=True,
        use_dropout=False,
        dropout_rate=0.5,
        out_dim=500,
        l2_reg=1e-4,
):

    # 输入层
    inputs = Input(shape=input_shape)
    x = inputs

    x = Conv1D(
        filters=conv_filters[0],
        kernel_size=kernel_sizes[0],
        padding='same',
        kernel_regularizer=l2(l2_reg)
    )(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(
        filters=conv_filters[1],
        kernel_size=kernel_sizes[1],
        padding='same',
        kernel_regularizer=l2(l2_reg)
    )(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2)(x)

    if use_eca:
        eca = ECABlock1D()(x)
        x = layers.add([x, eca])

    x = LSTM(lstm_units)(x)

    if use_dense and dense_units:
        for i, units in enumerate(dense_units, 1):
            x = Dense(
                units=units,
                activation='relu',
                kernel_regularizer=l2(l2_reg),
            )(x)

            if use_dropout:
                x = Dropout(dropout_rate)(x)

    outputs = Dense(out_dim, activation='linear')(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model







