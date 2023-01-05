import cv2
import numpy as np
from keras.models import model_from_json
from PIL import ImageFont, ImageDraw, Image

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# load json and create model
json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("model/emotion_model.h5")
print("Loaded model from disk")


# pass here your video path
# you may download one from here : https://www.pexels.com/video/three-girls-laughing-5273028/
# read image file
# img = cv2.imread("C:\\tech\\Study\\FER\\testdata\\man-g4474f6873_1920.jpg")
# img = cv2.imread("C:\\tech\\Study\\FER\\testdata\\woman-g0f7d9994a_1920.jpg")
# img = cv2.imread("./testdata/sad-g878ea9d68_1920.jpg")
img = cv2.imread("testdata/KakaoTalk_20230105_160716426.jpg")

frame = img
frame = cv2.resize(frame, (1280, 720))
face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# detect faces available on camera
num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
print(f"num_faces: {num_faces}")

# take each face available on the camera and Preprocess it
for (x, y, w, h) in num_faces:
    cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 4)
    roi_gray_frame = gray_frame[y:y + h, x:x + w]
    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

    # predict the emotions
    emotion_prediction = emotion_model.predict(cropped_img)
    print(f"emotion_prediction: {emotion_prediction}")
    text = ""

    for i, v in enumerate(emotion_prediction[0]):
        # v = v * 100
        text += f"{emotion_dict[i]}:{v}\n"

    y0, dy = 50, 25
    for i, line in enumerate(text.split('\n')):
        ny = y0 + i*dy
        cv2.putText(frame, line, (x+w+5, ny), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 0, 0), 2, cv2.LINE_AA)

# cv2.imwrite('res.jpg', frame)
cv2.imshow('Emotion Detection', frame)
cv2.waitKey(0)

