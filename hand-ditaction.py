import math
import time
import numpy as np
import cv2
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
cv2.namedWindow("Hand Filter", cv2.WINDOW_NORMAL)

# Initialize detector with reliable confidence threshold
detector = HandDetector(maxHands=2, detectionCon=0.5)

filters          = ["None", "GRAY", "THERMAL", "INVERT", "SKETCH", "VINTAGE"]
current_filter   = 0
last_switch_time = 0
cooldown         = 0.7

# Define a neat, classic font style
FONT_STYLE = cv2.FONT_HERSHEY_DUPLEX

def apply_filter(frame, name):
    """Return a fully-filtered copy of frame."""
    if name == "GRAY":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    elif name == "THERMAL":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    elif name == "INVERT":
        return cv2.bitwise_not(frame)

    elif name == "SKETCH":
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        inv     = 255 - gray
        blur    = cv2.GaussianBlur(inv, (21, 21), 0)
        invblur = 255 - blur
        sketch  = cv2.divide(gray, invblur, scale=256.0)
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

    elif name == "VINTAGE":
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        return np.clip(cv2.transform(frame, kernel), 0, 255).astype(np.uint8)

    return frame.copy()

def apply_filter_single_hand_box(frame, lmList, w, h, filter_name):
    """Apply filter ONLY in the box between the thumb (4) and index (8) of ONE hand."""
    tx, ty = lmList[4][:2]
    ix, iy = lmList[8][:2]

    x1, y1 = min(tx, ix), min(ty, iy)
    x2, y2 = max(tx, ix), max(ty, iy)

    padding = 25
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    if (x2 - x1) < 10 or (y2 - y1) < 10:
        return frame.copy()

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    filtered = apply_filter(frame, filter_name)
    mask3 = cv2.merge([mask, mask, mask])
    inside = cv2.bitwise_and(filtered, mask3)
    outside = cv2.bitwise_and(frame, cv2.bitwise_not(mask3))
    result = cv2.add(inside, outside)

    cv2.rectangle(result, (x1, y1), (x2, y2), (255, 0, 255), 2)
    return result

def apply_filter_in_box(frame, hand1_pts, hand2_pts, w, h, filter_name):
    """Apply filter inside the quadrilateral box spanned between TWO hands."""
    tl = hand1_pts[4][:2]
    tr = hand2_pts[4][:2]
    br = hand2_pts[8][:2]
    bl = hand1_pts[8][:2]

    pts = np.array([tl, tr, br, bl], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    filtered = apply_filter(frame, filter_name)
    mask3   = cv2.merge([mask, mask, mask])
    inside  = cv2.bitwise_and(filtered, mask3)
    outside = cv2.bitwise_and(frame, cv2.bitwise_not(mask3))
    result  = cv2.add(inside, outside)

    cv2.polylines(result, [pts], isClosed=True, color=(255, 0, 255), thickness=2)
    return result

skeleton_connections = [
    (0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8),
    (9,10), (10,11), (11,12), (13,14), (14,15), (15,16),
    (0,17), (17,18), (18,19), (19,20), (5,9), (9,13), (13,17)
]

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    hands, _ = detector.findHands(frame, draw=False)

    if hands:
        for hand in hands:
            lmList = hand["lmList"]

            for start, end in skeleton_connections:
                cv2.line(frame, tuple(lmList[start][:2]), tuple(lmList[end][:2]), (0, 255, 0), 2)
            for lm in lmList:
                cv2.circle(frame, tuple(lm[:2]), 5, (0, 0, 255), -1)

            tx, ty = lmList[4][0], lmList[4][1]
            ix, iy = lmList[8][0], lmList[8][1]
            now = time.time()
            dist_index = math.hypot(ix - tx, iy - ty)

            if dist_index < 65 and (now - last_switch_time) > cooldown:
                current_filter = (current_filter + 1) % len(filters)
                last_switch_time = now

    selected = filters[current_filter]

    # Dynamic Box Rendering Setup
    if len(hands) == 2:
        frame = apply_filter_in_box(frame, hands[0]["lmList"], hands[1]["lmList"], w, h, selected)
        status_text = "STATUS: Dual Hand Box"
    elif len(hands) == 1:
        frame = apply_filter_single_hand_box(frame, hands[0]["lmList"], w, h, selected)
        status_text = "STATUS: Finger Box Active"
    else:
        frame = apply_filter(frame, selected)
        status_text = "STATUS: Searching Hands..."

    # ------------------ CLASSIC NEAT UI LAYOUT ------------------
    # Top HUD Banner (Status & Active Filter Selection)
    cv2.rectangle(frame, (0, 0), (w, 45), (15, 15, 15), -1)
    cv2.putText(frame, status_text, (20, 28), FONT_STYLE, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FILTER: {selected}", (w - 220, 28), FONT_STYLE, 0.55, (255, 255, 0), 1, cv2.LINE_AA)

    # Bottom HUD Banner (Clean Interactive Instructions)
    cv2.rectangle(frame, (0, h - 40), (w, h), (15, 15, 15), -1)
    cv2.putText(frame, "CONTROLS: Pinch Fingers / Press 'N' to Cycle Filters | 'Q' to Quit", 
                (20, h - 14), FONT_STYLE, 0.48, (240, 240, 240), 1, cv2.LINE_AA)
    # ------------------------------------------------------------

    cv2.imshow("Hand Filter", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('n') or key == ord('N'):
        current_filter = (current_filter + 1) % len(filters)
    elif key == ord('q') or key == ord('Q'):
        break

cap.release()
cv2.destroyAllWindows()