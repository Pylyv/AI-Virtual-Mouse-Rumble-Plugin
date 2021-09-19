import cv2
import numpy
import numpy as np

import hands_detector as hd
import time
import pyautogui

###################################
w_cam, h_cam = 640, 480
frame_reduction = 150
smoothening = 2
###################################

prev_time = 0

# Tracks the needed values to smooth the mouse movement
prev_location_x, prev_location_y = 0, 0
current_location_x, current_location_y = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, w_cam)
cap.set(4, h_cam)
detector = hd.HandsDetector(max_hands=1)

w_screen, h_screen = pyautogui.size()
print(w_screen, h_screen)

while True:
    # 1. Find hand landmarks
    success, img = cap.read()
    img = detector.find_hands(img)
    landmark_list, bbox = detector.find_position(img)

    # 2. Get the tip of the index and middle fingers
    if len(landmark_list) != 0:
        x1, y1 = landmark_list[8][1:]
        x2, y2 = landmark_list[12][1:]
        # 3. Check which fingers are up
        fingers_up = detector.fingers_up()
        cv2.rectangle(
            img,
            (frame_reduction, frame_reduction),
            (w_cam - frame_reduction, h_cam - frame_reduction),
            (255, 0, 255), 2
        )

        # 4. Only index finger: Moving move
        if fingers_up[1] == 1 and fingers_up[2] == 0:
            # 5. Convert coordinates
            x3 = np.interp(x1, (frame_reduction, w_cam-frame_reduction), (0, w_screen))
            y3 = np.interp(y1, (frame_reduction, h_cam-frame_reduction), (0, h_screen))

            # 6. Smoothen values
            current_location_x = prev_location_x + (x3 - prev_location_x) / smoothening
            current_location_y = prev_location_y + (y3 - prev_location_y) / smoothening

            # 7. Move mouse
            pyautogui.moveTo(w_screen - current_location_x, current_location_y)
            cv2.circle(img, (x1, y1,), 15, (255, 0, 255), cv2.FILLED)
            prev_location_x, prev_location_y = current_location_x, current_location_y

        # 8. Both index and middle fingers are up: Clicking mode
        if fingers_up[1] == 1 and fingers_up[2] == 1:
            # 9. Find distance between fingers
            length, img, line_info = detector.find_distance(8, 12, img)
            print(length)
            if length < 40:
                cv2.circle(img, (line_info[4], line_info[5]), 15, (0, 255, 0), cv2.FILLED)
                # 10. Click mouse if distance short
                pyautogui.click()

    # 11. Frame rate
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(
        img, str(int(fps)), (20, 50),
        cv2.FONT_HERSHEY_PLAIN, 3,
        (255, 0, 0), 3
    )
    # 12. Display
    cv2.imshow('Image', img)
    cv2.waitKey(1)
