import cv2
import time
import numpy as np
import hands_detector as hd
import math

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

#############################
weight_cam, height_cam = 640, 480
#############################

cap = cv2.VideoCapture(0)
cap.set(3, weight_cam)
cap.set(4, height_cam)
previous_time = 0

detector = hd.HandsDetector(detection_confidence=0.7)


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volume_range = volume.GetVolumeRange()
minimum_volume = volume_range[0]
maximum_volume = volume_range[1]
vol = 0
vol_bar = 400
vol_percentage = 0

while True:
    success, img = cap.read()
    img = detector.find_hands(img)
    landmark_list = detector.find_position(img, draw=False)
    if len(landmark_list) != 0:

        x1, y1 = landmark_list[4][1], landmark_list[4][2]
        x2, y2 = landmark_list[8][1], landmark_list[8][2]
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (center_x, center_y), 15, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)

        # Hand range 50 - 300
        # Volume range -65 -0

        vol = np.interp(length, [50, 300], [minimum_volume, maximum_volume])
        vol_bar = np.interp(length, [50, 300], [400, 150])
        vol_percentage = np.interp(length, [50, 300], [0, 100])
        volume.SetMasterVolumeLevel(vol, None)

        if length < 50:
            cv2.circle(img, (center_x, center_y), 15, (0, 255, 0), cv2.FILLED)

    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 255), 3)
    cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, f'{int(vol_percentage)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)

    current_time = time.time()
    fps = 1/(current_time - previous_time)
    previous_time = current_time

    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    cv2.imshow("Img", img)
    cv2.waitKey(1)