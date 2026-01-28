from keras.api.layers import Input, Dense, Dropout
from keras.api.models import Model
from keras.api.regularizers import l2


def  build_mlp_model(
    input_shape=(700,),
    hidden_units=[256, 128, 64],
    out_dim=1,
    dropout_rate=0.5,
    l2_reg=0.0
 ):

    inputs = Input(shape=input_shape, name="input")
    x = inputs
    for units in hidden_units:
        x = Dense(
            units=units,
            activation='relu',
            kernel_regularizer=l2(l2_reg)
        )(x)

        if dropout_rate and dropout_rate > 0:
            x = Dropout(dropout_rate)(x)

    outputs = Dense(out_dim, activation='linear')(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model







