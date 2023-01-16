from flask import Flask, render_template, request, send_file
import cv2
import numpy as np
from keras.models import model_from_json
from io import BytesIO
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


def get_res(file):
    # 감정 class
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    # 이미지 파일 로드, cv2 포맷으로 디코드
    img = file.read()
    img = cv2.imdecode(np.fromstring(img, np.uint8), cv2.IMREAD_UNCHANGED)
    # 이미지 사이즈 조절 및 gray 스케일 전환
    frame = img
    # frame = cv2.resize(frame, (1024, 768))
    frame = cv2.resize(frame, (1280, 720))
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 모델 로드
    json_file = open('model/emotion_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    emotion_model = model_from_json(loaded_model_json)
    # 모델 weight 로드
    emotion_model.load_weights("model/emotion_model.h5")

    # 얼굴 감지 객체 로드
    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    # 감지된 얼굴 객체
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # 감지된 얼굴 객체 -> 모델을 통한 감정 분석 -> 이미지에 결과 그리기
    for (x, y, w, h) in num_faces:
        # 사각형 그리기
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # 얼굴 부분만 추출
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # 얼굴 -> 모델, 감정 분석
        emotion_prediction = emotion_model.predict(cropped_img)
        print(f"emotion_prediction: {emotion_prediction}")

        # 감정 -> 이미지에 그리기
        text = ""
        dy = 30
        for i, v in enumerate(emotion_prediction[0]):
            text += f"{emotion_dict[i]}:{round(float(v), 4)}\n"
        for i, line in enumerate(text.split('\n')):
            ny = y + i * dy
            cv2.putText(frame, line, (x + w + 5, ny), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    # 결과, cv2 포멧 -> jpg 형식으로 인코딩
    _, result = cv2.imencode('.jpg', frame)
    # ndarray -> 파일 객체로 형변환
    result = BytesIO(result)
    return result


@app.route("/")
def main():
    # templates 폴더에서 탐색된 파일을 기준
    return render_template('index.html')


@app.route('/analysed-emotions', methods=['POST'])
def predict():
    try:
        file = request.files['image']
        result = get_res(file)
        return send_file(result, mimetype='image/jpeg')
    except Exception as e:
        print(e)
        # templates 폴더에서 탐색된 파일을 기준
        return render_template('error.html')


if __name__ == "__main__":
    app.run(debug=True)
