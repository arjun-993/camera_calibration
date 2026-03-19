import cv2

cap = cv2.VideoCapture(0)

count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1)

    if key == ord('s'):
        filename = "images/chess" + str(count) + ".jpg"
        cv2.imwrite(filename, frame)
        print("Saved:", filename)
        count += 1

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()