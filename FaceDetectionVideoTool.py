# -*- coding: utf-8 -*-
import numpy as np
import cv2

# Загрузка предварительно обученной модели для распознавания лиц
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# Функция для обнаружения и отрисовки прямоугольников вокруг лиц
def detect_and_draw_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=6, minSize=(60, 60))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return frame


# Загрузка видео
video_path = '1.mp4'  # Замените на путь к вашему видео
cap = cv2.VideoCapture(video_path)

# проверка открытия видео
if not cap.isOpened():
    print("Ошибка: Не удалось открыть видео.")
    exit()

# Получение fps и размера кадра из исходного видео
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output.mp4", fourcc, fps, (frame_width, frame_height))


try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Достигнут конец видео.")
            break

        frame_with_faces = detect_and_draw_faces(frame)

        # Запись кадра в файл
        out.write(frame_with_faces)

        cv2.imshow('Video', frame_with_faces)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except Exception as e:
    print(f"Произошла ошибка: {e}")

cap.release()
out.release()
cv2.destroyAllWindows()
