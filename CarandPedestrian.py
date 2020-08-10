import cv2

# Images
#Insert image of cars here
img_file = 'Car Image.jpg'

# Video 
#Insert video of dash cam videos here
video = cv2.VideoCapture("DashCam.mp4")

# Pre-Trained Car Classifer
classifier_file_car = 'car_detector.xml'
classifier_file_pedestrian = 'pedestrian.xml'

# Create a car classifier
car_tracker = cv2.CascadeClassifier(classifier_file_car)

# Create a pedestrian classifier
pedestrian_tracker = cv2.CascadeClassifier(classifier_file_pedestrian)

while True:

    # Read the current frame
    (read_successful, frame) = video.read()

    
    if read_successful:
        # Convert to grayscale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # Detect Cars and Pedestrian
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrian = pedestrian_tracker.detectMultiScale(grayscaled_frame)

    print(cars)

    # Draw rectangles around the cars and pesdestrian
    # [x, y, w, h]       x+w y+h             color    thickness
    for (x, y, w,h) in cars:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0 , 255), 2)

    for (x, y, w,h) in pedestrian:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 255), 2)
    
    # Display the images with the face spotted
    cv2.imshow('Car Detector', frame)


    # Dont autoclose (Wait here in the code and listen for a key press)
    cv2.waitKey(1)

    # Stop if Q key is pressed
    if key==81 or key == 113:
        break

# Release the VideoCapture object
video.release()

"""
# Create an opencv image
img = cv2.imread(img_file)

# convert to greyscale (needed for haar casacade)
black_and_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Create a car classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

# Detect Cars
cars = car_tracker.detectMultiScale(black_and_white)

print(cars)


#car1 = cars[1]
#(x,y,w,h) = car1


# Draw rectangles around the cars
# [x, y, w, h]       x+w y+h             color    thickness
for (x, y, w,h) in cars:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0 ), 2)

# Display the images with the face spotted
#cv2.imshow('Car Detector', black_and_white)
cv2.imshow('Car Detector', img)


# Dont autoclose (Wait here in the code and listen for a key press)
cv2.waitKey()
"""

print("Code Completed")