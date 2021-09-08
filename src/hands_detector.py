import cv2
import mediapipe as mp
import time


class HandsDetector:
    def __init__(self, mode=False, max_hands=2, detection_confidence=0.5, track_confidence=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.track_confidence = track_confidence

        self.results = None

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands, self.detection_confidence, self.track_confidence)

        self.mp_draw = mp.solutions.drawing_utils

        self.tip_ids = [4, 8, 12, 16, 20]

        self.landmarks_list = []

    def find_hands(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if hands_detected := self.results.multi_hand_landmarks:
            for hand_landmarks in hands_detected:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, hand_number=0, draw=True):

        self.landmarks_list = []

        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_number]

            for id, ld in enumerate(my_hand.landmark):
                height, width, channels = img.shape
                position_x, position_y = int(ld.x * width), int(ld.y * height)

                self.landmarks_list.append([id, position_x, position_y])

        return self.landmarks_list

    def fingers_up(self):
        fingers = []

        # thumb
        if self.landmarks_list[self.tip_ids[0]][1] > self.landmarks_list[self.tip_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # for fingers
        for id in range(1, 5):
            if self.landmarks_list[self.tip_ids[id]][2] < self.landmarks_list[self.tip_ids[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers