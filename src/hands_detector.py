import cv2
import mediapipe as mp
import math
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

        x_list = []
        y_list = []
        bounding_box = []
        self.landmarks_list = []

        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_number]

            for id, ld in enumerate(my_hand.landmark):
                height, width, channels = img.shape
                position_x, position_y = int(ld.x * width), int(ld.y * height)
                x_list.append(position_x)
                y_list.append(position_y)
                self.landmarks_list.append([id, position_x, position_y])

                if draw:
                    cv2.circle(img, (position_x, position_y), 5, (255, 0, 255), cv2.FILLED)
            x_min, x_max = min(x_list), max(x_list)
            y_min, y_max = min(y_list), max(y_list)
            bounding_box = x_min, y_min, x_max, y_max

            if draw:
                cv2.rectangle(img, (bounding_box[0] - 20, bounding_box[1] - 20),
                              (bounding_box[2] + 20, bounding_box[3] + 20), (0, 255, 0), 2)
        return self.landmarks_list, bounding_box

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

    def find_distance(self, point1, point2, img, draw=True):

        x1, y1 = self.landmarks_list[point1][1], self.landmarks_list[point1][2]
        x2, y2 = self.landmarks_list[point2][1], self.landmarks_list[point2][2]
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (center_x, center_y), 15, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, center_x, center_y]