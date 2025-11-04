import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
import tensorflow as tf

from app.services.layer_distance import L1Distance

model = tf.keras.models.load_model('model/siamesemodel.h5', custom_objects={'L1Distance': L1Distance}, compile=False)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss=tf.keras.losses.BinaryCrossentropy())
embedding_model = model.layers[2]
l1_model = model.layers[3]
dense_model = model.layers[4]

embeddings_db = np.load('application_data/embeddings/embeddings_db.npy', allow_pickle=True).item()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (105, 105)) / 255.0
        face_tensor = tf.expand_dims(face, axis=0)
        
        face_embedding = embedding_model(face_tensor)
        
        best_name = 'Desconhecido'
        best_score = -1
        
        for name, ref_embedding in embeddings_db.items():
            ref_embedding_tensor = tf.convert_to_tensor(ref_embedding, dtype=tf.float32)
            dist = l1_model(face_embedding, ref_embedding_tensor)
            score = dense_model(dist)
            score = score.numpy().item()
            
            if score > best_score:
                best_score = score
                best_name = name
        
        threshold = 0.5
        if best_score < threshold:
            best_name = "Desconhecido"
            
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{best_name} ({best_score:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
    cv2.imshow('Reconhecimento Facial', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()