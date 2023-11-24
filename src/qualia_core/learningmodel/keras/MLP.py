def MLP(input_shape, output_shape, units: tuple=(100, 100, 100)):
    from tensorflow.keras.layers import Input, Dense, Flatten, Activation 
    from tensorflow.keras.models import Sequential

    model = Sequential()
    model.add(Input(input_shape))
    model.add(Flatten())
    for unit in units:
        model.add(Dense(unit))
        model.add(Activation('relu'))

    model.add(Dense(output_shape[0]))
    model.add(Activation('softmax'))
    
    return model
