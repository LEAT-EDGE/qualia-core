from typing import Union
from collections.abc import Iterable

def CNN(input_shape,
        output_shape,
        filters: tuple=(6, 16, 120),
        kernel_sizes: tuple=(3, 3, 5),
        paddings: tuple=(0, 0, 0),
        batch_norm: bool=False,
        dropouts: Union[float, list[float]] = 0.0,
        pool_sizes: tuple=(2, 2, 0),
        fc_units: int=(84, ),
        prepool: int=0,

        dims: int=1):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, Dropout, Activation, Input
    from tensorflow.keras.optimizers import Adadelta, SGD, Adam
    from tensorflow.keras.callbacks import LambdaCallback
    import tensorflow.keras.backend as backend

    if dims == 1:
        import qualia_core.learningmodel.keras.layers1d as layers_t
    elif dims == 2:
        import qualia_core.learningmodel.keras.layers2d as layers_t
    else:
        raise ValueError('Only dims=1 or dims=2 supported')

    # Backward compatibility for config not defining dropout as a list
    if not isinstance(dropouts, list):
        dropouts = [dropouts] * (len(filters) + len(fc_units))

    optimizer = Adam()

    model = Sequential()

    # Conv → MaxPool → Conv → MaxPool → Conv → Flatten → FullyConnected → FullyConnected(out)
    # No Dropout/BatchNormalization: doesn't improve test acc on UCI-HAR (even worse)

    model.add(Input(shape=input_shape))

    if prepool:
        model.add(layers_t.AveragePooling(pool_size=prepool))

    for f, ks, ps, padding, dropout in zip(filters, kernel_sizes, pool_sizes, paddings, dropouts):
        if isinstance(padding, Iterable) and sum(padding) > 0 or padding > 0:
            model.add(layers_t.ZeroPadding(padding=padding))
        model.add(layers_t.Conv(filters=f, kernel_size=ks))
        model.add(Activation('relu'))
        if batch_norm:
            model.add(BatchNormalization())
        if dropout:
            model.add(Dropout(dropout))
        if ps: # Optional MaxPool, must specify None in pool_sizes param when no MaxPool layer should be generated after Conv/Act
            model.add(layers_t.MaxPooling(pool_size=ps))

    model.add(Flatten())

    for fc_unit in fc_units:
        model.add(Dense(fc_unit))
        model.add(Activation('relu'))

    # Output classifier
    model.add(Dense(output_shape[0]))
    model.add(Activation('softmax'))

    return model
