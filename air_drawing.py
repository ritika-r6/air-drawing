import cv2
import numpy as np

# --- Color ranges in HSV ---
# Change these values to match your colored object (default: blue marker cap)
color_ranges = {
    "blue":   ([100, 150, 50], [140, 255, 255]),
    "green":  ([40,  70,  50], [80,  255, 255]),
    "red":    ([0,   150, 50], [10,  255, 255]),
    "yellow": ([20,  150, 50], [35,  255, 255]),
}

active_color = "blue"
lower = np.array(color_ranges[active_color][0])
upper = np.array(color_ranges[active_color][1])

# Drawing settings
draw_color = (255, 0, 0)   # BGR: blue
brush_size = 5
draw_colors_bgr = {
    "blue":   (255,   0,   0),
    "green":  (  0, 200,   0),
    "red":    (  0,   0, 220),
    "yellow": (  0, 220, 220),
}

# --- Setup ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Canvas to draw on (black background, same size as frame)
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

prev_x, prev_y = 0, 0

print("=== Air Drawing System ===")
print("Hold a colored object (e.g. blue marker cap) in front of the camera.")
print("Keys: b=blue  g=green  r=red  y=yellow  c=clear  q=quit")
print("      +/- to change brush size")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Step 1: Flip frame (mirror effect)
    frame = cv2.flip(frame, 1)

    # Step 2: Convert BGR -> HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Step 3: Create mask for the selected color
    mask = cv2.inRange(hsv, lower, upper)

    # Step 4: Noise removal - erode then dilate (morphological ops)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Step 5: Find contours of the detected color
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Get the largest contour
        largest = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest) > 500:   # ignore tiny blobs
            # Step 5b: Calculate centroid from moments
            M = cv2.moments(largest)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Draw circle around detected object on frame
                x, y, w, h = cv2.boundingRect(largest)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 8, draw_color, -1)

                # Step 6: Draw line on canvas
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = cx, cy

                cv2.line(canvas, (prev_x, prev_y), (cx, cy), draw_color, brush_size)
                prev_x, prev_y = cx, cy
            else:
                prev_x, prev_y = 0, 0
    else:
        prev_x, prev_y = 0, 0

    # Step 7: Merge canvas with live frame for display
    # Convert canvas to grayscale mask so drawing shows on top
    canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, canvas_mask = cv2.threshold(canvas_gray, 10, 255, cv2.THRESH_BINARY)
    canvas_mask_inv = cv2.bitwise_not(canvas_mask)

    frame_bg = cv2.bitwise_and(frame, frame, mask=canvas_mask_inv)
    canvas_fg = cv2.bitwise_and(canvas, canvas, mask=canvas_mask)
    combined = cv2.add(frame_bg, canvas_fg)

    # Show the mask (for demo purposes - shows HSV detection)
    mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Add UI text on frame
    cv2.putText(combined, f"Color: {active_color}  Brush: {brush_size}px",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(combined, "b/g/r/y=color  c=clear  +/-=size  q=quit",
                (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    # Step 7: Display output windows
    cv2.imshow("Air Drawing - Output", combined)
    cv2.imshow("HSV Mask (Color Detection)", mask_colored)

    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        print("Canvas cleared.")
    elif key == ord('+') or key == ord('='):
        brush_size = min(brush_size + 2, 30)
        print(f"Brush size: {brush_size}")
    elif key == ord('-'):
        brush_size = max(brush_size - 2, 1)
        print(f"Brush size: {brush_size}")
    elif key in [ord('b'), ord('g'), ord('r'), ord('y')]:
        c = chr(key)
        mapping = {'b': 'blue', 'g': 'green', 'r': 'red', 'y': 'yellow'}
        active_color = mapping[c]
        lower = np.array(color_ranges[active_color][0])
        upper = np.array(color_ranges[active_color][1])
        draw_color = draw_colors_bgr[active_color]
        print(f"Color changed to: {active_color}")

cap.release()
cv2.destroyAllWindows()
