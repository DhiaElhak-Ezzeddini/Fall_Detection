import cv2
import numpy as np

# ---- Input / Output ----
input_path  = './Dataset/Fall/Raw_Video/S_M_81.mp4'
output_path = "output_test_flow.mp4"

cap = cv2.VideoCapture(input_path)

# Check input
ret, prev_frame = cap.read()
if not ret:
    print("ERROR: Cannot read first frame.")
    exit()

h, w = prev_frame.shape[:2]

# ---- Output video writer ----
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, 30, (w, h))



prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)



    # ===== 2. Optical Flow on foreground only =====
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray,
                                        None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)

    # magnitude and angle not needed, but can be useful
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    vis = frame.copy()

    # draw arrows only on FG mask
    step = 10
    for y in range(0, h, step):
        for x in range(0, w, step):
            dx = flow[y, x, 0]
            dy = flow[y, x, 1]
            cv2.arrowedLine(vis, (x, y),
                            (int(x + dx*5), int(y + dy*5)),
                            (0, 0, 255), 1, tipLength=0.3)

    # ===== 3. Save frame =====
    out.write(vis)

    # ===== 4. Display (optional) =====
    cv2.imshow("Foreground Mask", frame)
    cv2.imshow("Optical Flow on FG", vis)

    prev_gray = gray

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Saved:", output_path)
