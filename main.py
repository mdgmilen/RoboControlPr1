
import cv2
import mediapipe as mp
import time
from HandTrackingDynamic import HandTrackingDynamic

def main():
    print("<<<<<<<<<<Start1")
    ctime = 0  # Current time for FPS calculation
    ptime = 0  # Previous time for FPS calculation
    detector = HandTrackingDynamic()  # Initialize the hand tracking class
    print("<<<<<<<<<<Start2")
    detector.openCamera()
    while True:
        ret, frame = detector.cap.read()  # Read a frame from the webcam
        if not detector.isFrameOk(ret):  # If frame is not captured
            break
        frame = detector.findFingers(frame)  # Detect and draw hand landmarks
        result = detector.findPosition(frame)
        if result is not None:
            lmsList, _ = result  # Unpack the result
        command = detector.processMovement()  # Process movement and get command
        if command:  # If a command is detected
            cv2.putText(frame, command, (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)  # Display the command
            print(command)
        ctime = time.time()  # Get current time
        # fps = 1 / (ctime - ptime) if (ctime - ptime) > 0 else 0  # Calculate FPS
        # fps = 0 if (ctime - ptime) <= 0 else 1 / (ctime - ptime)
        if ((ctime - ptime) > 0):
            fps = 1 / (ctime - ptime)
        else:
            fps = 0
        ptime = ctime  # Update previous time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 120), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)  # Display FPS
        cv2.imshow('Hand Gesture Control', frame)  # Show the video frame
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Break loop on 'q' key press
            print("Quitting...")
            break
        #close if
    #end of while True:
    detector.closeCamera()

    print(">>>>>>>>>>>End")
main()