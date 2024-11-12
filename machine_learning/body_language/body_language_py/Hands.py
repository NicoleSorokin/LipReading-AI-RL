#region Importing Libraries
import cv2
import cv2.data
from cvzone.HandTrackingModule import HandDetector
import mediapipe as mp
import math
import dlib

#Load the ditector and predictor
#detector = dlib.get_frontal_face_detector()
#predictor = dlib.shape_predictor("/shape_predictor_68_face_landmarks.dat")

# Used to convert protobuf message to a dictionary
from google.protobuf.json_format import MessageToDict
#endregion

#Defines the bounds of the most recently detected face

#-------------------------------------------------------------------------------

def create_unit_vector(point1, point2):
    vector = [0,0,0]

    vector[0] = point2.x - point1.x
    vector[1] = point2.y - point1.y
    vector[2] = point2.z - point1.z
    mag = math.sqrt((vector[0] ** 2) + (vector[1] ** 2) + (vector[2] ** 2))
    vector = [vector[0]/mag,vector[1]/mag,vector[2]/mag]
    return vector

#-------------------------------------------------------------------------------

def compare_vector(point1a, point2a, point1b, point2b, tol):
    v1 = create_unit_vector(point1a, point2a)
    v2 = create_unit_vector(point1b, point2b)

    # Ignore z-values when flattening for straightness
    if v1[0] >= v2[0] - tol and  v1[0] <= v2[0] + tol:
        if v1[1] >= v2[1] - tol and  v1[1] <= v2[1] + tol:
            return True
        
    return False

#-------------------------------------------------------------------------------

def dist_check(point1, point2, dist):
    #Define Delta Values
    x = point2.x - point1.x
    y = point2.y - point1.y
    z = point2.z - point1.z

    # Determine if close 
    if (dist >= math.sqrt((x**2) + (y**2) + (z**2))):
        return True
    return False

#-------------------------------------------------------------------------------

def regular_read(hand_data, hand_straighten, hand_class):
    # Check Open Palm Positions; Upwards: Open, Calm | Downwards: Authority, Power, Negative
    if(sum(hand_straighten[1:]) == 4):
        comp_vector = create_unit_vector(hand_data[9],hand_data[12])
        if(comp_vector[1] > comp_vector[2]):
            if ((hand_class == "Left" and hand_data[2].x > hand_data[17].x) or (hand_class == "Right" and hand_data[2].x < hand_data[17].x)):
                return 0.5
            else:
                return -0.5
        else:
            if ((hand_class == "Left" and hand_data[2].x < hand_data[17].x) or (hand_class == "Right" and hand_data[2].x > hand_data[17].x)):
                return 0.5
            else:
                return -0.5
            
    # Check for closed fist  
    elif (sum(hand_straighten[1:]) == 0):        
        clench = []
        for i in range(0,4):
            clench.append(dist_check(hand_data[5 + (4 * i)], hand_data[8 + (4 * i)], 0.14))
        if sum(clench) == 4:
            return -0.75
        
    # Check for gestures
    elif (sum(hand_straighten[1:]) == 1):
        if hand_straighten[1]:
            return -0.25
        elif hand_straighten[2]:
            return -1
        return -0.1
    
    # Check for final gesture
    elif (sum(hand_straighten[1:]) == 2 and hand_straighten[1] and hand_straighten[2]):
        return 0.25
    
    return 0

#-------------------------------------------------------------------------------

def determineHands(results):
    # Obtain Hand Landmarks
    hand_data = []
    # Get left and right hand classification
    hand_class = []

    if results.multi_hand_landmarks:
        for hand in results.multi_handedness:
            # Image is flipped, hands are read in reverse
            if hand.classification[0].label == "Left":
                hand_class.append("Right")
            else:
                hand_class.append("Left")
        
        if len(results.multi_hand_landmarks) > 1:
            for handLandmarks in results.multi_hand_landmarks:
                for landmark in handLandmarks.landmark:
                    hand_data.append(landmark) #Take Hand Coords
        else:
            handLandmarks = results.multi_hand_landmarks[0]
            for landmark in handLandmarks.landmark:
                hand_data.append(landmark) #Take Hand Coords

    # Seperate into hands
    if len(hand_data) == 0:
        return [0,0,0]
    elif len(hand_data) > 21:
        hand_data = [hand_data[:21], hand_data[21:]]
    else:
        hand_data = [hand_data]

    # Straight Checker
    hand_straight = []
    for l in range(len(hand_data)):
        straightened = []
        for i in range(0,5):
            if compare_vector(hand_data[l][1 + (4 * i)], hand_data[l][4 + (4 * i)], hand_data[l][2 + (4 * i)], hand_data[l][3 + (4 * i)], 0.1):
                straightened.append(True)
            else:
                straightened.append(False)
        hand_straight.append(straightened)
    
    #Define the Return Value
    read_result = [0,0,0]
    first_read = False
    second_read = True

    if len(hand_data) > 1:
        # Ensure to read second hand
        second_read = False

        # Clenching, Steepling, else: read regularly
        if dist_check(hand_data[0][0], hand_data[1][0], 0.2):
            total = sum(hand_straight[0]) + sum(hand_straight[1])
            if total >= 5:
                total = (total - (10 - total)) * 0.05
            else:
                total = (total - 10) * 0.05

            return ["C", "C", round(total, 5)]
        
        steepled = []
        for i in range(1,6):
            steepled.append(dist_check(hand_data[0][4 * i], hand_data[1][4 * i], 0.1))

        if sum(steepled) >= 1:
            return ["S","S", round(sum(steepled) / 10, 5)]
    
    #if one hand on face
    #if other hand on face

    if not first_read:
        read_result[0] = regular_read(hand_data[0], hand_straight[0], hand_class[0])

    if not second_read:
        read_result[1] = regular_read(hand_data[1], hand_straight[1], hand_class[1])

    read_result[2] = sum(read_result)

    for i in range(0,3):
        if type(read_result[i]) == float:
            read_result[i] = round(read_result[i], 5)

    return read_result

#-------------------------------------------------------------------------------

def detect_faces(image):
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(grey_image, 1.1, 5, minSize = (40, 40))

    '''
    faceL = detector(grey_image, 1)
    print(faceL)
    
    for face in faceL:
        landmarks = predictor(grey_image, face)
        print(landmarks)
    '''

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 4)
    
    return (faces, 0)

#-------------------------------------------------------------------------------
# Body Language Detection

#region Initializing the Model
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    max_num_hands=2)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
#endregion

# Open the capture and set dimensions
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

#Detect Hand Regions
hDetector = HandDetector(detectionCon=0.8, maxHands=2)

# Set Face Tracking
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand annotations on the image
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                
        # Flip the image horizontally for a selfie-view display
        cv2.flip(image, 1)

        # Read Body Language
        if True:
            # Get Face Data
            faces = detect_faces(image)

            # Get Hand Region and Z Coordinates
            handBounds = []
            hand_z = []
            list_hands = hDetector.findHands(image, draw=False)

            for hand in list_hands[0]:
                if hand != []:
                    x1, y1 = hand['lmList'][5][0], hand['lmList'][5][1]
                    x2, y2 = hand['lmList'][17][0], hand['lmList'][17][1]
                    distance = int(math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2))
                    print(distance)

                    handBounds.append(list(hand['bbox']))
            
            # Draw Hand Bounds
            for rect in handBounds:
                cv2.rectangle(image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255,0,0))
            
            reading = determineHands(results)
            cv2.putText(image, "First Hand:", (5,15), 4, 0.5, (0,0,0))
            cv2.putText(image, "Second Hand:", (5,30), 4, 0.5, (0,0,0))
            cv2.putText(image, "Total:", (5,45), 4, 0.5, (0,0,0))
            cv2.putText(image, str(reading[0]), (150,15), 4, 0.5, (0,0,0))
            cv2.putText(image, str(reading[1]), (150,30), 4, 0.5, (0,0,0))
            cv2.putText(image, str(reading[2]), (150,45), 4, 0.5, (0,0,0))
        
        else:
            print("Fail")
            cv2.putText(image, "First Hand:", (5,15), 4, 0.5, (0,0,0))
            cv2.putText(image, "Second Hand:", (5,30), 4, 0.5, (0,0,0))
            cv2.putText(image, "Total:", (5,45), 4, 0.5, (0,0,0))

        # Display Results
        cv2.imshow('Hand-Body Language Reader', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()



            