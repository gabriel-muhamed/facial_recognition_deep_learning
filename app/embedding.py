from keras.layers import Conv2D, Dense, MaxPooling2D, Input, Flatten
from keras.models import Model

class Embedding:
    def __init__(self):
        _inp = Input(shape=(105,105,3))
        
        _c1 = Conv2D(64, (10, 10), activation='relu')(_inp)
        _m1 = MaxPooling2D(64, (2, 2), padding='same')(_c1)
        
        _c2 = Conv2D(128, (7,7), activation='relu')(_m1)
        _m2 = MaxPooling2D(64, (2, 2), padding='same')(_c2)
        
        _c3 = Conv2D(128, (4,4), activation='relu')(_m2)
        _m3 = MaxPooling2D(64, (2, 2), padding='same')(_c3)
        
        _c4 = Conv2D(256, (4,4), activation='relu')(_m3)
        _f1 = Flatten()(_c4)
        _d1 = Dense(4096, activation='sigmoid')(_f1)
        
        self.model = Model(inputs=_inp, outputs=_d1, name='embedding')
    
    def __call__(self, inp):
        return self.model(inp)