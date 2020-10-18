import cv2
from time import sleep
import shelve

# for recording live eye data to an external shelve file in order to use
# the live data independently
eye_info = shelve.open("eye_info")

cascade_path = "haarcascade_lefteye_2splits.xml"
# cascade_path1 = "haarcascade_frontalface_default.xml"
# face_cascade = cv2.CascadeClassifier(cascade_path1)
eye_cascade = cv2.CascadeClassifier(cascade_path)
cam = cv2.VideoCapture(0)
anterior = 0
while True:
    if not cam.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass
    # Capture frame-by-frame
    ret, frame = cam.read()
    frame = frame [0:3000, 0:1500]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))
    # Draw a rectangle around the faces
    # for (x, y, w, h) in faces:
    #    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    count = 0
    for (xe, ye, we, he) in eyes:
        if count < 2:
            # bounding eye
            cv2.rectangle(frame, (xe, ye), (xe+we, ye+he), (255, 0, 0), 2)
            count += 1
            # writing eye's position, width, and height
            eye_info["x"] = xe
            eye_info["y"] = ye
            eye_info["width"] = we
            eye_info["height"] = he
            # bounding pupil
            roi = frame[ye: ye+he, xe: xe+we]
            rows, cols, _ = roi.shape
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray_roi = cv2.GaussianBlur(gray_roi, (25, 25), 0)
            _, threshold = cv2.threshold(gray_roi, 60, 255, cv2.THRESH_BINARY_INV)
            contours, dcba = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
            for cnt in contours:
                (x, y, w, h) = cv2.boundingRect(cnt)
                cv2.drawContours(roi, [cnt], -1, (0, 0, 255), 2)
                # cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.line(roi, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 1)
                cv2.line(roi, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 1)
                break
        else:
            break
    # Display the resulting frame
    # cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Display the resulting frame
    cv2.imshow('Tracker', frame)
# When everything is done, release the capture
cam.release()
cv2.destroyAllWindows()
