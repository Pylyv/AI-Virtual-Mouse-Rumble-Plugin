import cv2
import numpy as np
import time
import os
import hands_detector as hd

##############################
brush_thickness = 15
eraser_thickness = 50
#############################

folder_path = "../header"
my_list = os.listdir(folder_path)

overlay_list = []
for image_path in my_list:
    image = cv2.imread(f'{folder_path}/{image_path}')
    overlay_list.append(image)

header = overlay_list[0]
h, w, c = header.shape
draw_color = (255, 0, 255)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = hd.HandsDetector(detection_confidence=0.85)

x_previous, y_previous = 0, 0

img_canvas = np.zeros((720, 1280, 3), np.uint8)
while True:

    # 1. Import image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # 2. Find Hand Landmarks
    img = detector.find_hands(img)
    landmark_list = detector.find_position(img, draw=False)

    if len(landmark_list) != 0:
        # Tip of index and middle finger
        x1, y1 = landmark_list[8][1:]
        x2, y2 = landmark_list[12][1:]

        # 3. Check which fingers are up
        fingers = detector.fingers_up()

        # 4. If selection mode - 2 fingers are up
        if fingers[1] and fingers[2]:
            x_previous, y_previous = 0, 0
            # checking for the click
            if y1 < h:
                if 250 < x1 < 450:
                    header = overlay_list[0]
                    draw_color = (255, 0, 255)
                elif 550 < x1 < 750:
                    header = overlay_list[1]
                    draw_color = (255, 0, 0)
                elif 800 < x1 < 950:
                    header = overlay_list[2]
                    draw_color = (0, 255, 0)
                elif 1050 < x1 < 1200:
                    header = overlay_list[3]
                    draw_color = (0, 0, 0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), draw_color, cv2.FILLED)

        # 5. If drawing mode - index finger is up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, draw_color, cv2.FILLED)

            if x_previous == 0 and y_previous == 0:
                x_previous, y_previous = x1, y1

            if draw_color == (0, 0, 0):
                cv2.line(img, (x_previous, y_previous), (x1, y1), draw_color, eraser_thickness)
                cv2.line(img_canvas, (x_previous, y_previous), (x1, y1), draw_color, eraser_thickness)
            else:
                cv2.line(img, (x_previous, y_previous), (x1, y1), draw_color, brush_thickness)
                cv2.line(img_canvas, (x_previous, y_previous), (x1, y1), draw_color, brush_thickness)

            x_previous, y_previous = x1, y1

    # adding the 2 images together (img and img_canvas)
    img_gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
    _, img_inverse = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_inverse = cv2.cvtColor(img_inverse, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, img_inverse)
    img = cv2.bitwise_or(img, img_canvas)

    # Setting the header image
    h, w, c = header.shape
    img[0:h, 0:w] = header

    # img = cv2.addWeighted(img, 0.5, img_canvas, 0.5, 0)

    cv2.imshow("Image", img)
    # cv2.imshow("Canvas", img_canvas)
    cv2.waitKey(1)