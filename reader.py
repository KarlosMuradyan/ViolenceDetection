import cv2
import time as ts


class VideoReader:
    def __init__(self, video_file_path, fps):
        if video_file_path == "0" or video_file_path == "webcam":
            video_file_path = 0 # for webcam
            print('reading from webcam:')
        self.file_path = video_file_path
        self.fps = fps
        self.cap = None
        self.step = 1. / self.fps
        self.second = -self.step

    def __iter__(self):
        if not self.cap:
            self.cap = cv2.VideoCapture(self.file_path)
        while True:
            ret, frame = self.cap.read()

            if not ret:
                break

            self.second += self.step
            yield (frame, self.second)

class VideoStreamReader:
    def __init__(self, stream_url, fps):
        self.url = stream_url
        self.fps = fps
        self.cap = None
        self.step = 1. / self.fps
        self.second = -self.step

    def __iter__(self):
        if not self.cap:
            self.cap = cv2.VideoCapture(self.url)

        while True:
           # ts.sleep(self.step)  # in case of 25 fps 0.04 -> 40 milliseconds

            ret, frame = self.cap.read()

            if not ret:
                break

            self.second += self.step
            yield (frame, self.second)

        self.cap.release()
