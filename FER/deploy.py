from flask import Flask, render_template, request, send_file
import cv2
import numpy as np
from tensorflow import keras
from io import BytesIO
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


def get_res(file):
    # 이미지 파일 로드, cv2 포맷으로 디코드
    img = file.read()
    img = cv2.imdecode(np.fromstring(img, np.uint8), cv2.IMREAD_UNCHANGED)
    # 이미지 사이즈 조절 및 gray 스케일 전환
    img = cv2.resize(img, (1280, 720))
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 감지된 얼굴 객체
    faces = face_detector.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5)

    # 각 얼굴에 대한 감정을 추론하고 이미지에 표시
    for (x, y, w, h) in faces:
        # 얼굴 영역 추출
        face_in_gray_img = gray_img[y:y+h, x:x+w]
        # 차원 수정 (배치 크기, 너비, 높이, 채널) = (1, 48, 48, 1)
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(face_in_gray_img, (48, 48)), -1), 0)
        # 감정 추론
        predicted_emotion = model.predict(cropped_img)
        
        # 기존 이미지에 박스표시
        cv2.rectangle(img, (x, y-50), (x + w, y + h + 10), (255, 255, 255), 2)
        # 각 감정에 대한 모델 값을 text에 저장
        text = ""
        for i, v in enumerate(predicted_emotion[0]):
            v = round(v * 100)
            text += f"{emotion_dict[i]:}: {v}%\n"
        y0, dy = 50, 25
        for i, line in enumerate(text.split('\n')):
            ny = y0 + i*dy
            cv2.putText(
                img=img,
                text=line,
                org=(x+w+5, ny),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.7,
                color=(255,255,255),
                lineType=cv2.LINE_AA
            )

    # 결과, cv2 포멧 -> jpg 형식으로 인코딩
    _, result = cv2.imencode('.jpg', img)
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
    # 감정 class
    emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprise"}
    # 모델 로드
    model = keras.models.load_model("my_model.h5")
    # 얼굴 감지 객체 로드
    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

    app.run(debug=True)
