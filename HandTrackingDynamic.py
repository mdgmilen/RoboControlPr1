import cv2
import mediapipe as mp
import time
# import absl.logging

class HandTrackingDynamic:
    def __init__(self, mode=False, maxHands=1, detectionCon=0.5, trackCon=0.5):
        # absl.logging.set_verbosity(absl.logging.INFO)
        # absl.logging.use_absl_handler()
        self.cap = None
        self.__mode__ = mode
        self.__maxHands__ = maxHands
        self.__detectionCon__ = detectionCon
        self.__trackCon__ = trackCon
        self.handsMp = mp.solutions.hands
        self.hands = self.handsMp.Hands(static_image_mode=mode, max_num_hands=maxHands,
                                        min_detection_confidence=detectionCon, min_tracking_confidence=trackCon
                                        # , image_size=(640, 480)
                                        )
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.startPoint = None
        self.endPoint = None
        self.movementCommand = None

    def findFingers(self, frame, draw=True):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape  # Get the dimensions of the frame
        self.results = self.hands.process(cv2.resize(imgRGB, (w, h)))
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLms, self.handsMp.HAND_CONNECTIONS)
        return frame

    def findPosition(self, frame, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmsList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmsList.append([id, cx, cy])
                # print(self.lmsList[0])
                if draw:
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(frame, (xmin - 20, ymin - 20), (xmax + 20,    ymax + 20), (0, 255, 0), 2)

    def openCamera(self):
        self.cap = cv2.VideoCapture(0)
        print("<<<<<<<<<<Start3")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set video width
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set video height
        if not self.cap.isOpened():  # Check if the camera opened successfully
            print("Cannot open camera")  # Print error message
            exit()  # Exit the program
        else:
            print("Camera opened.")  # Print error message

    def closeCamera(self):
        self.cap.release()  # Release the webcam
        cv2.destroyAllWindows()  # Close all OpenCV windows

    def isFrameOk(self, ret):
        isOk = True
        if not ret:  # If frame is not captured
            print("Failed to grab frame")  # Print error message
            isOk = False  # Exit the loop
        return isOk

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
        thresholdSMovement = 50  # Threshold for significant movement
        if abs(dx) > abs(dy):  # Horizontal movement
            if dx > thresholdSMovement:  # Threshold for significant movement
                return "TURN LEFT"  # Command: Move Left
            elif dx < -thresholdSMovement:
                return "TURN RIGHT"  # Command: Move Right
        else:  # Vertical movement
            if dy > thresholdSMovement:
                return "MOVE BACKWARDS"  # Command: Move Backwards
            elif dy < -thresholdSMovement:
                return "MOVE STRAIGHT"  # Command: Move Straight
        return "STILL STOP"  # No significant movement

    def processMovement(self):
        # Process the movement and detect commands
        fingers = self.findFingerUp()  # Get finger states
        if sum(fingers) == 0:  # All fingers down (fist)
            self.movementCommand = "STOP"  # Command: Stop
            self.startPoint = None  # Reset start point
            self.endPoint = None  # Reset end point
        elif self.lmsList:  # If landmarks are detected
        # if sum(fingers) == 4:  # All fingers down (fist)
            palmX, palmY = self.lmsList[0][1:]  # Palm center (landmark 0)
            if not self.startPoint:  # If no start point
                self.startPoint = (palmX, palmY)  # Set start point
            else:  # If start point exists
                # time.sleep(0.3)
                self.endPoint = (palmX, palmY)  # Update end point
                self.movementCommand = self.detectCommand()  # Detect command
        return self.movementCommand  # Return the movement command