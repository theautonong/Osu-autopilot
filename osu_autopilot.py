# osu hacks without tensorflow
import time
import random
import threading
import tkinter as tk
import numpy as np
import cv2
from mss import mss
import pyautogui

# Classes
class PerformanceTracker:
    def __init__(self):
        self.total_clicks = 0
        self.successful_clicks = 0
        self.total_time = 0.0
        self._lock = threading.Lock()

    def log_click(self, success: bool):
        with self._lock:
            self.total_clicks += 1
            if success:
                self.successful_clicks += 1

    def log_time(self, time_taken: float):
        with self._lock:
            self.total_time += float(time_taken)

    def get_accuracy(self) -> float:
        with self._lock:
            return (self.successful_clicks / self.total_clicks) if self.total_clicks else 0.0

    def get_average_response_time(self) -> float:
        with self._lock:
            return (self.total_time / self.total_clicks) if self.total_clicks else 0.0


class QLearningAgent:
    def __init__(self, action_space, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.action_space = list(action_space)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}

    def _get_state_qs(self, state):
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in self.action_space}
        return self.q_table[state]

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.action_space)
        state_qs = self._get_state_qs(state)
        max_q = max(state_qs.values())
        best_actions = [a for a, q in state_qs.items() if q == max_q]
        return random.choice(best_actions)

    def update_q_table(self, state, action, reward, next_state):
        state_qs = self._get_state_qs(state)
        old_value = state_qs[action]
        next_qs = self._get_state_qs(next_state)
        next_max = max(next_qs.values()) if next_qs else 0.0
        state_qs[action] = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)


# Functions
def detect_circles(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                               param1=50, param2=30, minRadius=5, maxRadius=200)
    return np.round(circles[0, :]).astype("int").tolist() if circles is not None else []


def detect_sliders_and_spinners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sliders, spinners = [], []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:
            continue
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            sliders.append(approx)
        elif len(approx) > 8:
            spinners.append(cnt)
    return sliders, spinners


def plan_slider_path(slider_contour):
    pts = slider_contour.reshape(-1, 2)
    return [(int(x), int(y)) for x, y in pts]


def adaptive_click_with_precision(x, y, delay=0.05):
    try:
        start = time.time()
        pyautogui.moveTo(x, y, duration=0.08)
        time.sleep(delay)
        pyautogui.click()
        return time.time() - start
    except:
        return 0.0


# Interface
class PerformanceUI:
    def __init__(self, tracker):
        self.tracker = tracker
        self.root = tk.Tk()
        self.root.title("osu! Autopilot Performance")
        self.root.geometry("350x80")
        self.label = tk.Label(self.root, font=("TkDefaultFont", 10))
        self.label.pack(padx=10, pady=10)
        self._running = True
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        self._running = False
        self.root.quit()

    def update_ui(self):
        if not self._running:
            return
        acc = self.tracker.get_accuracy() * 100
        avg_rt = self.tracker.get_average_response_time()
        self.label.config(text=f"Accuracy: {acc:.2f}%, Avg Response Time: {avg_rt:.3f}s")
        self.root.after(500, self.update_ui)

    def run(self):
        self.update_ui()
        self.root.mainloop()


# loop
def autopilot_loop(tracker, stop_event):
    ql_agent = QLearningAgent(action_space=['click', 'hold', 'release'])
    sct = mss()
    monitor = {"top": 100, "left": 100, "width": 800, "height": 600}

    while not stop_event.is_set():
        loop_start = time.time()
        try:
            sct_img = np.array(sct.grab(monitor))
            img = cv2.cvtColor(sct_img, cv2.COLOR_BGRA2BGR)

            circles = detect_circles(img)
            sliders, spinners = detect_sliders_and_spinners(img)

            for circle in circles:
                x, y, _ = circle
                state = (x, y)
                action = ql_agent.get_action(state)
                if action == 'click':
                    time_taken = adaptive_click_with_precision(x + monitor["left"], y + monitor["top"], delay=0.04)
                    tracker.log_click(time_taken > 0)
                    tracker.log_time(time_taken)
                    ql_agent.update_q_table(state, action, 1.0 if time_taken > 0 else -0.2, state)

            for slider in sliders:
                path = plan_slider_path(slider)
                for px, py in path:
                    state = (px, py)
                    action = ql_agent.get_action(state)
                    if action == 'click':
                        time_taken = adaptive_click_with_precision(px + monitor["left"], py + monitor["top"], delay=0.02)
                        tracker.log_click(time_taken > 0)
                        tracker.log_time(time_taken)
                        ql_agent.update_q_table(state, action, 1.0 if time_taken > 0 else -0.1, state)

            for spinner in spinners:
                x, y, w, h = cv2.boundingRect(spinner)
                cx, cy = x + w // 2 + monitor["left"], y + h // 2 + monitor["top"]
                pyautogui.moveTo(cx, cy, duration=0.08)
                pyautogui.mouseDown()
                time.sleep(1.5)
                pyautogui.mouseUp()
                tracker.log_click(True)
                tracker.log_time(1.5)

            cv2.imshow('Autopilot Preview', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break

            elapsed = time.time() - loop_start
            if elapsed < 0.01:
                time.sleep(0.01)

        except Exception as e:
            print(f"Error loop: {e}")
            time.sleep(0.2)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.0

    tracker = PerformanceTracker()
    ui = PerformanceUI(tracker)

    stop_event = threading.Event()
    threading.Thread(target=autopilot_loop, args=(tracker, stop_event), daemon=True).start()

    try:
        ui.run()
    finally:
        stop_event.set()
        # made by sailentcoder

