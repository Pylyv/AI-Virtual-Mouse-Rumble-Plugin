import cv2
import time
import os
import hands_detector as hd

###############################
width_cam, height_cam = 1280, 680
##############################

cap = cv2.VideoCapture(0)
cap.set(3, width_cam)
cap.set(4, height_cam)

folder_path = "../fingers"
my_list = os.listdir(folder_path)
overlay_list = []

for image_path in my_list:
    image = cv2.imread(f'{folder_path}/{image_path}')
    overlay_list.append(image)

previous_time = 0

detector = hd.HandsDetector(detection_confidence=0.75)

tip_ids = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()

    img = detector.find_hands(img)
    landmarks_list = detector.find_position(img, draw=False)

    if len(landmarks_list) != 0:
        fingers = []

        # thumb
        if landmarks_list[tip_ids[0]][1] > landmarks_list[tip_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # for fingers
        for id in range(1, 5):
            if landmarks_list[tip_ids[id]][2] < landmarks_list[tip_ids[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        total_fingers = fingers.count(1)

    # h, w, c = overlay_list[0].shape -----> do this if your images have different sizes
        img[0:250, 0:250] = overlay_list[total_fingers - 1]

        cv2.rectangle(img, (20, 250), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(total_fingers), (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)

    current_time = time.time()
    fps = 1/(current_time - previous_time)
    previous_time = current_time

    cv2.putText(img, f'FPS: {int(fps)}', (950, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)