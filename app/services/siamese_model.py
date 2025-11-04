from keras.layers import Input, Dense
from keras.models import Model

class SiameseModel:
    def __init__(self, embedding, layer_distance):
        self._embedding = embedding
        self._layer_distance = layer_distance
    
    def __call__(self):
        input_image = Input(name='input_img', shape=(105,105,3))
        validation_image = Input(name='validation_img', shape=(105,105,3))
        
        siamese_layer = self._layer_distance
        siamese_layer.__name__ = 'distance'
        distances = siamese_layer(self._embedding(input_image), self._embedding(validation_image))
        
        classifier = Dense(1, activation='sigmoid')(distances)
        
        return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')