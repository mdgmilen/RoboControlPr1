#%%
import cv2  # Import OpenCV for video capturing and image processing
import mediapipe as mp  # Import Mediapipe for hand tracking
import time  # Import time for calculating FPS
import math  # Import math for calculations
#%%
class HandTrackingDynamic:
    def __init__(self, mode=False, maxHands=1, detectionCon=0.5, trackCon=0.5):
        # Initialize Mediapipe Hands with parameters for detection and tracking
        self.__mode__ = mode  # Static or dynamic mode for hand detection
        self.__maxHands__ = maxHands  # Maximum number of hands to detect
        self.__detectionCon__ = detectionCon  # Minimum detection confidence
        self.__trackCon__ = trackCon  # Minimum tracking confidence
        self.handsMp = mp.solutions.hands  # Mediapipe Hands solution
        self.hands = self.handsMp.Hands(static_image_mode=mode, max_num_hands=maxHands, 
                                        min_detection_confidence=detectionCon, min_tracking_confidence=trackCon)
        self.mpDraw = mp.solutions.drawing_utils  # Utility to draw hand landmarks
        self.tipIds = [4, 8, 12, 16, 20]  # IDs of fingertips for thumb and fingers
        self.startPoint = None  # To store the starting point of a gesture
        self.endPoint = None  # To store the ending point of a gesture
        self.movementCommand = None  # To store the detected movement command

    def findFingers(self, frame, draw=True):
        # Process the frame to detect hands
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB
        self.results = self.hands.process(imgRGB)  # Process the frame with Mediapipe Hands
        if self.results.multi_hand_landmarks:  # If hands are detected
            for handLms in self.results.multi_hand_landmarks:  # Iterate through detected hands
                if draw:  # If drawing is enabled
                    self.mpDraw.draw_landmarks(frame, handLms, self.handsMp.HAND_CONNECTIONS)  # Draw hand landmarks
        return frame  # Return the processed frame

    def findPosition(self, frame, handNo=0, draw=True):
        # Find the position of hand landmarks
        xList = []  # List to store x-coordinates of landmarks
        yList = []  # List to store y-coordinates of landmarks
        bbox = []  # Bounding box around the hand
        self.lmsList = []  # List to store landmark IDs and their coordinates

        if self.results.multi_hand_landmarks:  # If hands are detected
            myHand = self.results.multi_hand_landmarks[handNo]  # Get the specified hand
            for id, lm in enumerate(myHand.landmark):  # Iterate through landmarks
                h, w, c = frame.shape  # Get frame dimensions
                cx, cy = int(lm.x * w), int(lm.y * h)  # Convert normalized coordinates to pixel values
                xList.append(cx)  # Append x-coordinate to the list
                yList.append(cy)  # Append y-coordinate to the list
                self.lmsList.append([id, cx, cy])  # Store the landmark ID and coordinates
                print(self.lmsList[0])

                if draw:  # If drawing is enabled
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)  # Draw circles at landmarks

            xmin, xmax = min(xList), max(xList)  # Find min and max x-coordinates
            ymin, ymax = min(yList), max(yList)  # Find min and max y-coordinates
            bbox = xmin, ymin, xmax, ymax  # Define bounding box

            if draw:  # If drawing is enabled
                cv2.rectangle(frame, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                              (0, 255, 0), 2)  # Draw bounding box

        return self.lmsList, bbox  # Return landmark list and bounding box

    def findFingerUp(self):
        # Determine which fingers are up
        fingers = []  # List to store finger states
        if self.lmsList:  # If landmarks are available
            handType = "Right Hand" if self.lmsList[0][1] > self.lmsList[1][1] else "Left Hand"
            # Check thumb (left or right based on position)
            if self.lmsList[self.tipIds[0]][1] > self.lmsList[self.tipIds[0] - 1][1]:
                fingers.append(1)  # Thumb is up
            else:
                fingers.append(0)  # Thumb is down

            # Check other fingers (vertical position)
            for id in range(1, 5):
                if self.lmsList[self.tipIds[id]][2] < self.lmsList[self.tipIds[id] - 2][2]:
                    fingers.append(1)  # Finger is up
                else:
                    fingers.append(0)  # Finger is down
        return fingers  # Return list of finger states

    def detectCommand(self):
        # Detect the command based on start and end points
        if not self.startPoint or not self.endPoint:  # If either point is missing
            return None  # No command

        dx = self.endPoint[0] - self.startPoint[0]  # Calculate x-axis movement
        dy = self.endPoint[1] - self.startPoint[1]  # Calculate y-axis movement

        if abs(dx) > abs(dy):  # Horizontal movement
            if dx > 20:  # Threshold for significant movement
                return "MOVE LEFT"  # Command: Move Left
            elif dx < -20:
                return "MOVE RIGHT"  # Command: Move Right
        else:  # Vertical movement
            if dy > 20:
                return "MOVE BACKWARDS"  # Command: Move Backwards
            elif dy < -20:
                return "MOVE STRAIGHT"  # Command: Move Straight

        return None  # No significant movement

    def processMovement(self):
        # Process the movement and detect commands
        fingers = self.findFingerUp()  # Get finger states

        if sum(fingers) == 0:  # All fingers down (fist)
            self.movementCommand = "STOP"  # Command: Stop
            self.startPoint = None  # Reset start point
            self.endPoint = None  # Reset end point
        elif self.lmsList:  # If landmarks are detected
            palmX, palmY = self.lmsList[0][1:]  # Palm center (landmark 0)

            if not self.startPoint:  # If no start point
                self.startPoint = (palmX, palmY)  # Set start point
            else:  # If start point exists
                self.endPoint = (palmX, palmY)  # Update end point
                self.movementCommand = self.detectCommand()  # Detect command

        return self.movementCommand  # Return the movement command
#%%
def main():
    ctime = 0  # Current time for FPS calculation
    ptime = 0  # Previous time for FPS calculation
    cap = cv2.VideoCapture(0)  # Open the webcam
    detector = HandTrackingDynamic()  # Initialize the hand tracking class
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set video width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set video height

    if not cap.isOpened():  # Check if the camera opened successfully
        print("Cannot open camera")  # Print error message
        exit()  # Exit the program

    while True:  # Loop to process video frames
        ret, frame = cap.read()  # Read a frame from the webcam
        if not ret:  # If frame is not captured
            print("Failed to grab frame")  # Print error message
            break  # Exit the loop

        frame = detector.findFingers(frame)  # Detect and draw hand landmarks
        lmsList, _ = detector.findPosition(frame)  # Get landmark positions
        command = detector.processMovement()  # Process movement and get command

        if command:  # If a command is detected
            cv2.putText(frame, command, (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)  # Display the command

        ctime = time.time()  # Get current time
        fps = 1 / (ctime - ptime) if (ctime - ptime) > 0 else 0  # Calculate FPS
        ptime = ctime  # Update previous time

        cv2.putText(frame, f"FPS: {int(fps)}", (10, 120), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)  # Display FPS

        cv2.imshow('Hand Gesture Control', frame)  # Show the video frame
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Break loop on 'q' key press
            break

    cap.release()  # Release the webcam
    cv2.destroyAllWindows()  # Close all OpenCV windows
    
#%%
if __name__ == "__main__":
    main()  # Run the main function