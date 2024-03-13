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

    results = model(video_path)

    frames = []
    for i, frame in enumerate(results.render()):
        filename = f'frame_{i}.jpg'
        filepath = os.path.join(SAVE_DIR,filename)

        for box, label, conf in zip(results.xyxy[0], results.names[0], results.pred[0][:, 4]):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'({label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imwrite(filepath, frame)
        frames.append(filepath)

    video_capture = cv2.VideoCapture(video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)  # 동영상 프레임 속도
    frame_interval = int(fps * 2)  # 2초마다 프레임 추출
    count = 0  #

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        count += 1
        if count % frame_interval == 0:
            filename = f'frame_videio_{count}.jpg'
            filepath = os.path.join(SAVE_DIR, filename)
            cv2.imwrite(filepath, frame)
            frames.append(filepath)
    video_capture.release()

    return jsonify(frames)

    """
    fps = cap.get(cv2.CAP_PROP_FPS)  # 동영상 프레임 속도
    frame_interval = int(fps * 2)  # 2초마다 프레임 추출
    
    ## count = 0  # 수정해야함
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        ##count += 1
        # 2초마다 프레임을 추출하고 yolo모델에 프레임을 입력값으로 넘겨 객체탐지한다.
        # 객체 탐지 후 output은 results에 저장된다.
        ##if count % frame_interval == 0:

            results = model(frame)
            # results 에 담긴 정보를 추출한다.
            for box, label, conf in zip(results.xyxy[0], results.names[0], results.pred[0][:, 4]):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'({label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            ## filename = f'frame_{count}.jpg'  ## 수정해야함 파일명
            ## filepath = os.path.join(SAVE_DIR, filename)
            cv2.imwrite(filepath, frame)
            frames.append(filepath)
    cap.release()

    return jsonify(frames)
    """


if __name__ == '__main__':
    app.run(debug=True)
