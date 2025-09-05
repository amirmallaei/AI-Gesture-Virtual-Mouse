# virtual_mouse.py

import cv2
import mediapipe as mp
from ultralytics import YOLO
import pyautogui
import numpy as np
import math
import time
from typing import Tuple, Optional, Dict, Any

class GestureController:
    """
    A class to control the mouse and system functions using hand gestures
    captured from a webcam. It integrates YOLO for person detection and
    MediaPipe for detailed hand landmark detection.
    """
    def __init__(self):
        # --- Configuration ---
        self.SCREEN_W, self.SCREEN_H = pyautogui.size()
        self.CAM_W, self.CAM_H = 1280, 720
        self.FRAME_REDUCTION = 100  # Virtual border for mouse mapping

        # --- Smoothing & Cooldowns ---
        self.SMOOTHENING = 7
        self.CLICK_COOLDOWN = 0.5
        self.SCREENSHOT_COOLDOWN = 3.0
        self.CLOSE_COOLDOWN = 3.0
        self.ZOOM_CHANGE_THRESHOLD = 0.02 # Sensitivity for zoom

        # --- Model Initialization ---
        print("Loading models...")
        self.yolo_model = YOLO("yolov8n.pt")
        self.person_class_id = list(self.yolo_model.names.keys())[list(self.yolo_model.names.values()).index('person')]
        
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=2,
            min_detection_confidence=0.7, min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        print("Models loaded successfully.")

        # --- State Variables ---
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, self.CAM_W)
        self.cap.set(4, self.CAM_H)
        
        self.prev_x, self.prev_y = 0, 0
        self.last_action_times = {'click': 0, 'screenshot': 0, 'close': 0}
        self.initial_zoom_distance = None

    def _recognize_gesture(self, hand_landmarks: Any) -> Optional[str]:
        """
        Recognizes a specific gesture from a single hand's landmarks.
        
        Args:
            hand_landmarks: A MediaPipe hand_landmarks object.
        
        Returns:
            A string representing the recognized gesture, or None.
        """
        landmarks = hand_landmarks.landmark
        
        # Check if each finger is extended
        thumb_up = landmarks[4].x < landmarks[3].x < landmarks[2].x or landmarks[4].x > landmarks[3].x > landmarks[2].x
        index_up = landmarks[8].y < landmarks[6].y
        middle_up = landmarks[12].y < landmarks[10].y
        ring_up = landmarks[16].y < landmarks[14].y
        pinky_up = landmarks[20].y < landmarks[18].y
        
        # Define specific gestures based on which fingers are up
        if not (thumb_up or index_up or middle_up or ring_up or pinky_up): return "FIST"
        if thumb_up and index_up and middle_up and ring_up and pinky_up: return "PALM"
        if index_up and middle_up and not ring_up and not pinky_up: return "SCROLL_MODE"
        if index_up and not middle_up and not ring_up and not pinky_up: return "MOUSE_MODE"
        
        # Independent click gesture
        click_distance = math.hypot(landmarks[4].x - landmarks[8].x, landmarks[4].y - landmarks[8].y)
        if click_distance < 0.04: return "CLICK"

        return None

    def _execute_action(self, gesture: str, hand_data: Dict[str, Any], frame_shape: Tuple[int, ...]) -> None:
        """
        Executes a system action based on the recognized gesture.
        
        Args:
            gesture: The string name of the gesture to perform.
            hand_data: A dictionary containing landmark and position data.
            frame_shape: The shape of the camera frame.
        """
        current_time = time.time()
        landmarks = hand_data['landmarks']
        x1_p, y1_p = hand_data['person_box_origin']

        if gesture == "MOUSE_MODE":
            cam_x = int(landmarks[8].x * hand_data['crop_shape'][1]) + x1_p
            cam_y = int(landmarks[8].y * hand_data['crop_shape'][0]) + y1_p
            
            screen_x = np.interp(cam_x, (self.FRAME_REDUCTION, self.CAM_W - self.FRAME_REDUCTION), (0, self.SCREEN_W))
            screen_y = np.interp(cam_y, (self.FRAME_REDUCTION, self.CAM_H - self.FRAME_REDUCTION), (0, self.SCREEN_H))
            
            curr_x = self.prev_x + (screen_x - self.prev_x) / self.SMOOTHENING
            curr_y = self.prev_y + (screen_y - self.prev_y) / self.SMOOTHENING
            
            pyautogui.moveTo(curr_x, curr_y)
            self.prev_x, self.prev_y = curr_x, curr_y

        elif gesture == "SCROLL_MODE":
            wrist_y = int(landmarks[0].y * frame_shape[0])
            if 'prev_scroll_y' not in self.__dict__: self.prev_scroll_y = wrist_y
            
            if wrist_y < self.prev_scroll_y - 20: pyautogui.scroll(40)
            elif wrist_y > self.prev_scroll_y + 20: pyautogui.scroll(-40)
            self.prev_scroll_y = wrist_y

        elif gesture == "CLICK" and (current_time - self.last_action_times['click'] > self.CLICK_COOLDOWN):
            pyautogui.click()
            self.last_action_times['click'] = current_time

        elif gesture == "PALM" and (current_time - self.last_action_times['screenshot'] > self.SCREENSHOT_COOLDOWN):
            pyautogui.screenshot("screenshot.png")
            self.last_action_times['screenshot'] = current_time

        elif gesture == "FIST" and (current_time - self.last_action_times['close'] > self.CLOSE_COOLDOWN):
            pyautogui.hotkey('alt', 'f4')
            self.last_action_times['close'] = current_time

    def _handle_zoom(self, hand1_lm: Any, hand2_lm: Any) -> None:
        """Handles two-handed zoom gesture."""
        wrist1_x, wrist1_y = hand1_lm[0].x, hand1_lm[0].y
        wrist2_x, wrist2_y = hand2_lm[0].x, hand2_lm[0].y
        current_zoom_distance = math.hypot(wrist2_x - wrist1_x, wrist2_y - wrist1_y)
        
        if self.initial_zoom_distance is None:
            self.initial_zoom_distance = current_zoom_distance
        
        if current_zoom_distance > self.initial_zoom_distance + self.ZOOM_CHANGE_THRESHOLD:
            pyautogui.hotkey('ctrl', '+')
        elif current_zoom_distance < self.initial_zoom_distance - self.ZOOM_CHANGE_THRESHOLD:
            pyautogui.hotkey('ctrl', '-')
        
        self.initial_zoom_distance = current_zoom_distance

    def _draw_feedback(self, frame: np.ndarray, gesture: Optional[str]) -> None:
        """Draws visual feedback for the current gesture on the frame."""
        feedback_map = {
            "MOUSE_MODE": ("Mouse Mode", (0, 255, 0)),
            "SCROLL_MODE": ("Scroll Mode", (0, 255, 255)),
            "ZOOM_MODE": ("Zoom Mode", (255, 0, 255)),
            "CLICK": ("CLICK!", (255, 0, 255)),
            "PALM": ("Screenshot!", (0, 255, 0)),
            "FIST": ("Closing App!", (0, 0, 255)),
        }
        if gesture in feedback_map:
            text, color = feedback_map[gesture]
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

    def run(self):
        """
        Starts the main loop for camera capture, gesture recognition, and action execution.
        """
        print("--- Starting virtual mouse. Press 'q' to quit. ---")
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success: continue

            frame = cv2.flip(frame, 1)
            annotated_frame = frame.copy()
            
            yolo_results = self.yolo_model(frame, classes=[self.person_class_id], verbose=False)
            
            if yolo_results and len(yolo_results[0].boxes) > 0:
                person_box = yolo_results[0].boxes.xyxy.cpu().numpy().astype(int)[0]
                x1_p, y1_p, x2_p, y2_p = person_box
                person_crop = frame[y1_p:y2_p, x1_p:x2_p]

                if person_crop.size > 0:
                    person_crop_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                    mp_results = self.mp_hands.process(person_crop_rgb)

                    current_gesture = None
                    if mp_results.multi_hand_landmarks:
                        num_hands = len(mp_results.multi_hand_landmarks)

                        if num_hands == 2:
                            current_gesture = "ZOOM_MODE"
                            self._handle_zoom(mp_results.multi_hand_landmarks[0].landmark, mp_results.multi_hand_landmarks[1].landmark)
                        elif num_hands == 1:
                            self.initial_zoom_distance = None
                            hand_landmarks = mp_results.multi_hand_landmarks[0]
                            
                            current_gesture = self._recognize_gesture(hand_landmarks)
                            if current_gesture:
                                hand_data = {
                                    'landmarks': hand_landmarks.landmark,
                                    'person_box_origin': (x1_p, y1_p),
                                    'crop_shape': person_crop.shape
                                }
                                self._execute_action(current_gesture, hand_data, frame.shape)
                        
                        # Draw landmarks on all detected hands
                        for hand_landmarks in mp_results.multi_hand_landmarks:
                            self.mp_drawing.draw_landmarks(
                                image=annotated_frame[y1_p:y2_p, x1_p:x2_p],
                                landmark_list=hand_landmarks,
                                connections=mp.solutions.hands.HAND_CONNECTIONS
                            )
                    else:
                        self.initial_zoom_distance = None

                    self._draw_feedback(annotated_frame, current_gesture)
            
            cv2.imshow("Gesture Control", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # --- Clean up ---
        self.cap.release()
        cv2.destroyAllWindows()
        self.mp_hands.close()
        print("--- Virtual control stopped ---")


if __name__ == "__main__":
    controller = GestureController()
    controller.run()