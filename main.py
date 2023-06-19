import cv2
import numpy as np
from glob import glob
from pathlib import Path


#


class Face_detection:
    def __init__(self, webcam, image_path):
        # self.src = src
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "./haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "./haarcascade_eye.xml"
        )

        self.webcam = webcam
        self.image_path = image_path

    # Detect bounding box using cascade system
    def detection(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        facess = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        # print(facess)
        eye_boxes = []
        face_cord = []
        for x, y, w, h in facess:
            face_cord.append([x, y, w, h])
            roi_gray = gray[y : y + h, x : x + w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray)

            for ex, ey, ew, eh in eyes:
                eye_boxes.append((ex + x, ey + y, ew, eh))
        print(eye_boxes)
        return eye_boxes, face_cord

    # To vizualize the bbox on the images
    def vizualize(self, img, eye_boxes, faces_cord,capture):
        w,h,_ = img.shape
        size = (w,h)
        for i in range(len(eye_boxes)):
            left_eye = eye_boxes[i]
            img_annotate = cv2.rectangle(
                img,
                (left_eye[0], left_eye[1]),
                (left_eye[0] + left_eye[2], left_eye[1] + left_eye[3]),
                color=(0, 255, 0),
                thickness=2,
            )
        for j in range(len(faces_cord)):
            faces = faces_cord[j]
            print(faces)
            img_annotate = cv2.rectangle(
                img_annotate,
                (faces[0], faces[1]),
                (faces[0] + faces[2], faces[1] + faces[3]),
                color=(0, 0, 255),
                thickness=4,
            )
        if capture==True:
            result = cv2.VideoWriter('./output/vid1.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         20, size)
            result.write(img)
        cv2.namedWindow("img", cv2.WINDOW_NORMAL)

# Using resizeWindow()
        cv2.resizeWindow("img", 700, 200)
        cv2.imshow("img", img_annotate)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # This function works for images
    def image_detection(self):
        eyes = []
        faces = []
        j = 0  # denote the number of images
        for i in glob(self.image_path + "//*.jpg"):
            image = cv2.imread(i)
            a, b = self.detection(image)
            self.vizualize(image, a, b,False)
            eyes.append([f"{j} image", a])
            faces.append([f"{j} image", b])
            j += 1
        return eyes, faces

    # This function works for webcam feeding
    def video_detection(self, origin, path):
        eyes = []
        faces = []
        if origin == "webcam":
            live_stream = cv2.VideoCapture(0)
            while live_stream.isOpened():
                _, frame = live_stream.read()
                a, b = self.detection(frame)
                # print(a, b)
                if len(a) != 0 and len(b) != 0:
                    eyes.append(a)
                    faces.append(b)
                    self.vizualize(frame, a, b,True)
                else:
                    print("Either face or eyes may be not detected")
                    print("Please try again after removing the spec if you wearing one")
                    continue
            return eyes, faces
        else:
            vid = cv2.VideoCapture("./dataset/video.mp4")
            if vid.isOpened() == False:
                print("error in opening the file")
            while vid.isOpened():
                ret, frame = vid.read()
                if ret == True:
                    a, b = self.detection(frame)
                    eyes.append(a)
                    faces.append(b)
                    print(a, b)
                    if len(a) != 0 and len(b) != 0:
                        self.vizualize(frame, a, b,True)
                    else:
                        print("No face is detected")
                        continue
                    if cv2.waitKey(25) & 0xFF == ord("q"):
                        break
                else:
                    break
            return eyes, faces


def main(image_path, webcam, origin, path):
    detection = Face_detection(webcam, image_path)
    if webcam == "no":
        # This function activate the function that will work for webcam
        eyes, faces = detection.image_detection()
       
        return eyes, faces
    else:
        # This function activate the function that will work for images
        eyes, faces = detection.video_detection(origin, path)
        return eyes, faces


if __name__ == "__main__":
    webcam = "no"
    origin = "webcam"
    image_path = "./dataset"
    path = "./dataset/video.mp4"
    main(image_path, webcam, origin, path)
