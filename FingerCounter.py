import cv2
from cvzone.HandTrackingModule import HandDetector

# Initialize the hand detector
detector = HandDetector(maxHands=1, detectionCon=0.8)

# Initialize video capture
video = cv2.VideoCapture(0)  # Use 0 for the internal webcam, 1 for external webcam

while True:
    _, img = video.read()
    img = cv2.flip(img, 1)  # Flip the image for a mirror effect
    
    # Detect hands in the image
    hands, img = detector.findHands(img)  # returns the hands detected in the image and the image with drawn hand landmarks
    
    if hands:
        hand = hands[0]  # Only focus on the first detected hand
        finger_count = detector.fingersUp(hand)  # Get the number of fingers up (1 or 0 for each finger)
        num_fingers = sum(finger_count)  # Count the number of raised fingers
        
        # Display the number of fingers raised on the screen
        cv2.putText(img, f'Fingers: {num_fingers}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 4)
    
    # Show the image
    cv2.imshow("Finger Count", img)
    
    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
video.release()
cv2.destroyAllWindows()
