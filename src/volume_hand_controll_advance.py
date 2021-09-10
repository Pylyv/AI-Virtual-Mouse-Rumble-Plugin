import cv2
import time
import numpy as np
import hands_detector as hd
import math

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

#############################
width_cam, height_cam = 640, 480
#############################

cap = cv2.VideoCapture(0)
cap.set(3, width_cam)
cap.set(4, height_cam)
previous_time = 0

detector = hd.HandsDetector(detection_confidence=0.7, max_hands=1)


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
color_volume = (255, 0, 0)
area = 0

while True:
    success, img = cap.read()

    # Find Hand
    img = detector.find_hands(img)
    landmarks_list, bounding_box = detector.find_position(img, draw=True)
    if len(landmarks_list) != 0:

        # Filter based on size
        area = (bounding_box[2] - bounding_box[0]) * (bounding_box[3] - bounding_box[1]) // 100

        if 250 < area < 1000:
            # Find distance between index and thumb
            length, img, line_info = detector.find_distance(4, 8, img)

            # Convert volume
            vol_bar = np.interp(length, [50, 200], [300, 150])
            vol_percentage = np.interp(length, [50, 200], [0, 100])

            # Reduce resolution to make it smoother
            smoothness = 5
            vol_percentage = smoothness * round(vol_percentage / smoothness)

            # Check fingers up
            fingers = detector.fingers_up()

            # If pinky is down set volume
            if not fingers[4]:
                volume.SetMasterVolumeLevelScalar(vol_percentage / 100, None)
                cv2.circle(img, (line_info[4], line_info[5]), 15, (0, 255, 0), cv2.FILLED)
                color_volume = (0, 255, 0)
            else:
                color_volume = (255, 0, 0)

    # Drawings
    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 255), 3)
    cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, f'{int(vol_percentage)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)

    current_volume = int(volume.GetMasterVolumeLevelScalar() * 100)
    cv2.putText(img, f'Vol Set: {int(current_volume)}', (400, 50), cv2.FONT_HERSHEY_COMPLEX, 1, color_volume, 3)

    # Frame rate
    current_time = time.time()
    fps = 1/(current_time - previous_time)
    previous_time = current_time

    cv2.imshow("Img", img)
    cv2.waitKey(1)