import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model

# Load the face classifier and the emotion detection model
face_classifier = cv2.CascadeClassifier(r"C:\Deep Learning\haarcascade_frontalface_default.xml")
emotion_classifier = load_model(r"C:\Deep Learning\emotion_detection.h5")
gender_classifier = cv2.dnn.readNetFromCaffe(r"C:\Deep Learning\gender_deploy.prototxt", 
                                             r"C:\Deep Learning\gender_net.caffemodel")
age_classifier = cv2.dnn.readNetFromCaffe(r"C:\Deep Learning\age_deploy.prototxt",
                                          r"C:\Deep Learning\age_net.caffemodel")

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
gender_list = ['Male', 'Female']
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        # Emotion detection
        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = emotion_classifier.predict(roi)[0]
            emotion_label = emotion_labels[prediction.argmax()]
            label_position = (x, y - 10)
            cv2.putText(frame, emotion_label, label_position, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 165, 255), 2)
        
        # Gender and age detection
        face_blob = cv2.dnn.blobFromImage(roi_color, 1.0, (227, 227), (104.0, 177.0, 123.0), swapRB=False)
        
        # Predict gender
        gender_classifier.setInput(face_blob)
        gender_preds = gender_classifier.forward()
        gender = gender_list[gender_preds[0].argmax()]
        
        # Predict age
        age_classifier.setInput(face_blob)
        age_preds = age_classifier.forward()
        age = age_list[age_preds[0].argmax()]
        
        # Display gender and age
        label = f"{gender} , {age}"
        label_position = (x, y + h + 30)
        cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)
    
    cv2.imshow('Emotion, Age, and Gender Detector', frame)
    
    #Use the letter 'Q' from the keyboard to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

