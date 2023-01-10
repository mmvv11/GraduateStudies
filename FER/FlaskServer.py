# https://howtolivelikehuman.tistory.com/118

from flask import Flask, render_template, request, send_file
import cv2
import numpy as np
from keras.models import model_from_json
from io import BytesIO
from flask_cors import CORS



app = Flask(__name__)
CORS(app)

def get_res(file):

    # TODO: 모델 결과 돌려주기

    img = file.read()
    img =  cv2.imdecode(np.fromstring(img, np.uint8), cv2.IMREAD_UNCHANGED)

    ## 기존 코드
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    # load json and create model
    json_file = open('model/emotion_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    emotion_model = model_from_json(loaded_model_json)

    # load weights into new model
    emotion_model.load_weights("model/emotion_model.h5")
    # img = cv2.imread(file)
    frame = img
    frame = cv2.resize(frame, (1280, 720))
    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces available on camera
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

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
            ny = y0 + i * dy
            # cv2.rectangle(frame, (x + w + 5, ny), (x+10, ny+10), (0,0,0), -1)
            cv2.putText(frame, line, (x + w + 5, ny), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    _, result = cv2.imencode('.jpg', frame)
    # ndarray -> 파일 객체로 형변환
    result = BytesIO(result)
    return result


@app.route("/")
def main():
    # templates 폴더 안에서 찾음
    return render_template('index.html')



@app.route('/prediction', methods=['POST'])
def predict():
    try:
        file = request.files['image']
        result = get_res(file)
        print(type(result))
        return send_file(result, mimetype='image/jpeg')
    except Exception as e:
        print(e)
        return render_template('error.html')

if __name__ == "__main__":
    app.run(debug=True)