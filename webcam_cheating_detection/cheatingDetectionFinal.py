import os
import sys
import cv2
import numpy as np
import face_recognition
import math
import time
import csv
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import load_model


# --------------------------- STAGE 1: CALIBRATION ---------------------------

# 1a. Getting image of contestant.
"""We want to capture an image of the user, which will be used for facial 
recognition, to ensure that the user is not having someone else complete 
the mission for them."""
welcome_message = \
    """Welcome to the mission - to ensure a fair competition, we will be
      monitoring your activity through webcam.\n"""

print(welcome_message)

agreement = \
    """Do you agree to these terms? Type 'y' when you are ready 
    to begin the setup process.\n"""

agreed = input(agreement)

if (agreed != 'y'):
    print("""You did not agree to the terms, so you are not eligible to 
          participate in the contest.""")
    exit()

user_name = input("What is your full name?\n")
user_name = [name.capitalize() for name in user_name.split(" ")]
user_name = "".join(user_name)

screenshot_message = \
    """We will first need to capture a screenshot through your webcam to confirm 
    your identity. Please make sure you are seated in a place with good lighting 
    so that your face is clearly visible. Then, look straight into your webcam 
    and press 'c'.\n"""

print(screenshot_message)


""" The argument to VideoCapture depends how many webcam sources you have, if 
you don't have one hooked up it should be 0."""
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    sys.exit('Video source not found...')

while True:
    ret, frame = video_capture.read()
    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) == ord('c'):
        cv2.imwrite(f"faces/{user_name}.png", frame)
        video_capture.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        break




# 1b. Adjusting threshold parameter.
""" We want to adjust the threshold parameter needed for iris detection, 
which may depend on skin tone/lighting conditions of the individual's camera, 
so we will prompt the user to adjust the threshold to the correct position. """

webcam_message = \
      """In a moment, your webcam will turn on and you will see a threshold 
      bar. Please adjust the threshold until you see 2 circles appear on your 
      eyes. When this happens, we will capture a screenshot to ensure that 
      our cheating monitor is functioning properly. Press 'c' when you see 2
      circles around your irises. If the captured screenshot does not show
      proper iris detection, your submission may be disqualified.\n"""

print(webcam_message)
time.sleep(5)

"""Here is the initialization code and helper functions for the iris detection."""
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
detector = cv2.SimpleBlobDetector_create(detector_params)

# These parameters specify what level of eye movement is acceptable.
horizontal_boundary = 0.25
vertical_boundary = 0.25

def get_iris_centroid(keypoints):
    xcoords_sum = 0
    ycoords_sum = 0
    for keypoint in keypoints:
        xcoords_sum += keyPoint.pt[0]
        ycoords_sum += keyPoint.pt[1]
    
    num_points = len(keypoints)
    if num_points > 0:
        xcoords_sum /= num_points
        ycoords_sum /= num_points
    
    return xcoords_sum, ycoords_sum



def detect_faces(img, cascade):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = cascade.detectMultiScale(gray_frame, 1.3, 5)
    if len(coords) > 1:
        biggest = (0, 0, 0, 0)
        for i in coords:
            if i[3] > biggest[3]:
                biggest = i
        biggest = np.array([i], np.int32)
    elif len(coords) == 1:
        biggest = coords
    else:
        return None
    for (x, y, w, h) in biggest:
        frame = img[y:y + h, x:x + w]
    return frame


def detect_eyes(img, cascade):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = cascade.detectMultiScale(gray_frame, 1.3, 5)  # detect eyes
    width = np.size(img, 1)  # get face frame width
    height = np.size(img, 0)  # get face frame height
    left_eye = None
    right_eye = None
    left_coords = None
    right_coords = None
    for (x, y, w, h) in eyes:
        if y > height / 2:
            pass
        eyecenter = x + w / 2  # get the eye center
        if eyecenter < width * 0.5:
            left_eye = img[y:y + h, x:x + w]
            # The coordinates will be returned in the order: top, right, bottom, left
            left_coords = (y, x+w, y+h, x)
        else:
            right_eye = img[y:y + h, x:x + w]
            # The coordinates will be returned in the order: top, right, bottom, left
            right_coords = (y, x+w, y+h, x)
    return [left_eye, right_eye], left_coords, right_coords


def cut_eyebrows(img):
    height, width = img.shape[:2]
    eyebrow_h = int(height / 4)
    img = img[eyebrow_h:height, 0:width]  # cut eyebrows out (15 px)

    return img


def blob_process(img, threshold, detector):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)
    img = cv2.erode(img, None, iterations=2)
    img = cv2.dilate(img, None, iterations=4)
    img = cv2.medianBlur(img, 5)
    keypoints = detector.detect(img)
    #print(keypoints)
    return keypoints


def nothing(x):
    pass



# This will hold the specific threshold for the person; we set it to a default value here.
tailored_threshold = 80
accepted_capture = False

while not accepted_capture:
    """Now, we turn on the webcam and allow the user to perform calibration."""
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Calibration')
    cv2.createTrackbar('threshold', 'Calibration', 0, 255, nothing)
    while True:
        _, frame = cap.read()
        face_frame = detect_faces(frame, face_cascade)
        if face_frame is not None:
            eyes, _, _ = detect_eyes(face_frame, eye_cascade)
            for eye in eyes:
                if eye is not None:
                    threshold = r = cv2.getTrackbarPos('threshold', 'Calibration')
                    eye = cut_eyebrows(eye)
                    keypoints = blob_process(eye, threshold, detector)
                    eye = cv2.drawKeypoints(eye, keypoints, eye, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('Calibration', frame)
        if cv2.waitKey(1) == ord('c'):
            # We want to capture a screenshot, as well as record the threshold.
            tailored_threshold = cv2.getTrackbarPos('threshold', 'Calibration')
            cv2.imwrite(f"{user_name}_IrisCalibration.png", frame)
            cap.release()
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            break

    

    reminder_message = \
        """Here is the captured image. Keep in mind that both eyes should have 
        a circle imposed; otherwise, your submission may be disqualified. Do 
        you want to accept this image, or retake it? Type 'a' to accept or 'r'
        to retake.\n"""
    
    while True:
        captured_image = cv2.imread(f"{user_name}_IrisCalibration.png")
        cv2.imshow('Captured Image', captured_image)
        cv2.waitKey(1)
        retake = input(reminder_message)
        if retake == 'a':
            accepted_capture = True
            cv2.destroyAllWindows()
            break
        elif retake == 'r':
            cv2.destroyAllWindows()
            break
        else:
            cv2.destroyAllWindows()
            pass


# ------------------------ STAGE 2: FACIAL RECOGNITION ------------------------

"""We define a helper function as well as a class for facial recognition."""
def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)
    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'
    
class FaceRecognition:
    # Face locations stores the 4 coordinates for a tight bounding box around the face - top, right, bottom, left. 
    face_locations = []
    face_encodings = []
    face_names = []

    # These are the encodings for the images we provide the model with. 
    known_face_encodings = []

    # These are the image names we provide the model with. 
    known_face_names = []

    process_current_frame = True
    extensions = ["jpg", "jpeg", "png"]
    num_frames = 0

    def __init__(self):
        self.encode_faces()
    
    def encode_faces(self):
        for image in os.listdir('faces'):

            split_up = image.split(".")

            person_name = split_up[0]
            extension = split_up[-1].lower()

            print(f"{image}, {extension}")
            
            if extension not in self.extensions:
                continue


            face_image = face_recognition.load_image_file(f'faces/{image}')
            face_encoding = face_recognition.face_encodings(face_image)[0]


            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(person_name.capitalize())

        print(self.known_face_names)

    





    def run_recognition(self):
        # Depends how many webcam sources you have, if you don't have one hooked up it should be 0.
        video_capture = cv2.VideoCapture(0)
        cv2.namedWindow('Face Recognition and Iris Detection')

        # We won't be using a trackbar since we already have the threshold value from the calibration step.

        if not video_capture.isOpened():
            sys.exit('Video source not found...')
        while True:
            ret, frame = video_capture.read()

            if self.process_current_frame:

                # Decrease the image size by 1/4 in each axis. 
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

                # Standard format is BGR, convert to RGB.
                rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

                # Find all faces in current frame.
                self.face_locations = face_recognition.face_locations(rgb_small_frame)

                # Since we already know the location of the face, we can pass that in as a known face location. 
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

                # Check if there is more than one person in the frame. There should only be one.
                if len(self.face_encodings) > 1:
                    print("More than one person in frame, cheating detected!")

                if len(self.face_locations) == 0:
                    # No faces detected.
                    continue

                self.face_names = []
                for face_encoding in self.face_encodings:
                    # Compare the face in this frame to the faces that the model knows - returns a True/False list. 
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = "Unknown"
                    confidence = '???'

                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)

                    # We can't just use the above code since even the closest match might not be within the threshold, which is why we need the code below.
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = face_confidence(face_distances[best_match_index])

                        # Check if the person detected is different from who is supposed to be there.
                        if name != user_name:
                            #print("Another competitor has been detected in your frame!")
                            pass

                    # Check if we have detected an unknown person in the frame with high confidence.
                    else:
                        print("Unknown person in frame, cheating detected!")

                    self.face_names.append(f'{name} ({confidence})')
            self.process_current_frame = not self.process_current_frame

            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):

                top *= 4
                right *= 4
                bottom *= 4
                left *= 4


                # Cut out the face frame and make the image gray for the iris detection.
                cut_face = frame[top:bottom, left:right]

                # Perform eye detection to get the left and right eyes.
                eyes, left_coords, right_coords = detect_eyes(cut_face, eye_cascade)

                # Note that these eyes might be None types - do not attempt to perform calculations if this is the case.
                # Initialize the centroids for the irises.
                centroids = []
                
                if eyes[0] is None or eyes[1] is None:
                    # Eye detection is faulty, so we will not continue with analyzing this image.
                    #print("Iris detection returned None on one or more eyes.\n")
                    continue
                else:
                    # Process the eye images by cutting eyebrows.
                    for eye in eyes:
                        eye = cut_eyebrows(eye)
                        keypoints = blob_process(eye, tailored_threshold, detector)

                        # Now, we want to get the position of the center of the iris in relation to the box around the eyes.
                        x_center, y_center = get_iris_centroid(keypoints)
                        centroids.append([x_center, y_center])

                        # The 1st argument is the image from which keypoints are extracted; the 3rd argument is the image to draw them on.
                        eye = cv2.drawKeypoints(eye, keypoints, eye, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    
                    # Draw boxes around the eyes as well. 
                    lt, lr, lb, ll = left_coords
                    rt, rr, rb, rl = right_coords

                    cv2.rectangle(frame, (ll, lt), (lr, lb), (0, 0, 255), 2)
                    cv2.rectangle(frame, (rl, rt), (rr, rb), (0, 0, 255), 2)


                    """If the iris centroid is further left than the first quarter or further right than the 
                    last quarter, or if it is further up or down than those quarters, then we detect cheating
                    and take a snapshot. These are default values and can be changed."""
                    left_eye_width = lr - ll
                    right_eye_width = rr - rl
                    horizontal_cutoffs_left_eye = [ll + horizontal_boundary * left_eye_width, ll + (1 - horizontal_boundary * left_eye_width)]
                    horizontal_cutoffs_right_eye = [rl + horizontal_boundary * right_eye_width, rl + (1 - horizontal_boundary * right_eye_width)]

                    left_eye_height = lb - lt
                    right_eye_height = rb - rt
                    vertical_cutoffs_left_eye = [lb + vertical_boundary * left_eye_height, lb + (1 - vertical_boundary) * left_eye_height]
                    vertical_cutoffs_right_eye = [rb + vertical_boundary * right_eye_height, rb + (1 - vertical_boundary) * right_eye_height]
            
                    left_center_x, left_center_y = centroids[0]
                    right_center_x, right_center_y = centroids[1]

                    suspicious_movement_left = (left_center_x < horizontal_cutoffs_left_eye[0]) or (left_center_x > horizontal_cutoffs_left_eye[1]) \
                                            or (left_center_y < vertical_cutoffs_left_eye[0]) or (left_center_y > vertical_cutoffs_left_eye[1])
                    
                    suspicious_movement_right = (right_center_x < horizontal_cutoffs_right_eye[0]) or (right_center_x > horizontal_cutoffs_right_eye[1]) \
                                            or (right_center_y < vertical_cutoffs_right_eye[0]) or (right_center_y > vertical_cutoffs_right_eye[1])
                    
                    if suspicious_movement_left or suspicious_movement_right:
                        # We want to take a snapshot.
                        self.num_frames += 1
                        cv2.imwrite(f"suspicious/snapshot{self.num_frames}.png", frame)

                    

            cv2.imshow('Face Recognition and Iris Detection', frame)
            if cv2.waitKey(1) == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)


    
fr = FaceRecognition()
fr.run_recognition()



# -------------------------- STAGE 3: RUN CNN MODEL --------------------------

# Define some constants.
image_height= 480
image_width= 640

image_size = (224, 224)

# Put all suspicious screenshots into a csv file. 
with open(f"suspiciousScreenshots.csv", 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['ImageFileName'])
    for image in os.listdir("suspicious/"):
        writer.writerow([image])


# Load the saved model and use it to predict whether cheating occurred.
detect_cheating_model = load_model('vgg19.h5')

# Create a pandas dataframe for the suspicious screenshots.
check = pd.read_csv("suspiciousScreenshots.csv")

image_files = []
probabilities = []
predictions = []

for index, row in check.iterrows():
    img_name = row['ImageFileName']
    image_path = os.path.join("suspicious", img_name)

    split_up = img_name.split(".")

    person_name = split_up[0]
    extension = split_up[-1].lower()

    print(f"{img_name}, {extension}")
    
    if extension not in fr.extensions:
        continue

    # Load the image and preprocess it for prediction.
    img = load_img(image_path, target_size=image_size)
    img_array = img_to_array(img)

    # Keras models have the batch size as the first dimension so expand to create a batch size of 1. 
    img_array = np.expand_dims(img_array, axis=0)

    # Normalize the pixel values.
    img_array = img_array / 255.0

    # Make the prediction.
    prediction = detect_cheating_model.predict(img_array)

    binary_label = 1 if prediction[0][0] >= 0.5 else 0

    probabilities.append(prediction)
    image_files.append(img_name)
    predictions.append(binary_label)

print(list(zip(image_files, probabilities)))      
print(list(zip(image_files, predictions)))        
if sum(predictions) > 0:
    print("Cheating detected!")
