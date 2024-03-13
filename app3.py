import os
import cv2
import torch
import time

from flask import Flask, request, jsonify, render_template, redirect, url_for

SAVE_DIR = './saved_Detection'

app = Flask(__name__)

model_path = "./yolov5s.pt"

model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)


@app.route('/', methods=['GET', 'POST'])
def main_page():
    if request.method == 'POST' and 'video' in request.files:
        return redirect(url_for('upload_video'))
    return render_template("Main_page.html")


@app.route('/upload_video', methods=['POST'])
def upload_video():
    video_file = request.files['video']

    video_path = os.path.join(SAVE_DIR, 'uploaded_video.mp4')
    video_file.save(video_path)

    cap = cv2.VideoCapture(video_path)

    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)  # 동영상 프레임 속도

    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        count += 1
        results = model(frame)  # 각 프레임에 대해 객체 검출 수행

        filename = f'frame_{count}.jpg'
        filepath = os.path.join(SAVE_DIR, filename)

        # 바운딩 박스를 프레임 이미지에 그리고 저장
        for result in results.xyxy[0]:
            box = result[:4]  # 바운딩 박스의 좌표값 (x1, y1, x2, y2)
            label = result[5]  # 객체 클래스
            conf = result[4]  # 신뢰도
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'({label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imwrite(filepath, frame)
        frames.append(filepath)

    cap.release()

    return jsonify(frames)


if __name__ == '__main__':
    app.run(debug=True)
