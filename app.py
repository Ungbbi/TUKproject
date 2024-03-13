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
    if request.method == 'POST':
        if 'video' in request.files:
            return redirect(url_for('upload_video'))
        elif 'image' in request.files:
            return redirect(url_for('upload_image'))
    return render_template("Main_page.html")


@app.route('/upload_video', methods=['POST'])
def upload_video():
    video_file = request.files['video']

    video_path = os.path.join(SAVE_DIR, 'uploaded_video.mp4')
    video_file.save(video_path)

    cap = cv2.VideoCapture(video_path)

    frames = process_video(cap)

    return jsonify(frames)


@app.route('/upload_image', methods=['POST'])
def upload_image():
    image_file = request.files['image']

    image_path = os.path.join(SAVE_DIR, 'uploaded_image.jpg')
    image_file.save(image_path)

    image = cv2.imread(image_path)

    results = model(image)

    processed_image = draw_boxes(image, results)

    output_image_path = os.path.join(SAVE_DIR, 'processed_image.jpg')
    cv2.imwrite(output_image_path, processed_image)

    return jsonify(output_image_path)


def process_video(cap):
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)  # 동영상 프레임 속도

    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        count += 1
        results = model(frame)

        filename = f'frame_{count}.jpg'
        filepath = os.path.join(SAVE_DIR, filename)

        processed_frame = draw_boxes(frame, results)

        cv2.imwrite(filepath, processed_frame)
        frames.append(filepath)

    cap.release()
    return frames


def draw_boxes(image, results):
    for result in results.xyxy[0]:
        box = result[:4]  # 바운딩 박스의 좌표값 (x1, y1, x2, y2)
        label = result[5]  # 객체 클래스
        conf = result[4]  # 신뢰도
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f'({label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image


if __name__ == '__main__':
    app.run(debug=True)
