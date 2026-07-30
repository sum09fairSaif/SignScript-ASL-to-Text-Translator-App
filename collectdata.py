# Importing all required Python packages, libraries and frameworks
import cv2
import os

# Starting video capture
cap = cv2.VideoCapture(0)

# Making a directory to store the images
directory = 'Image/'

# Capturing the frames
while True:
    _, frame = cap.read()
    
    # Creating a dictionary to count the number of images in each subdirectory
    count = {
        'a': len(os.listdir(directory + '/A')),
        'b': len(os.listdir(directory + '/B')),
        'c': len(os.listdir(directory + '/C'))
    }
    
    # Getting the dimensions of each captured image
    row = frame.shape[1]
    col = frame.shape[0]

    # Drawing a white rectangle to show capture region
    cv2.rectangle(frame, (0, 40), (300, 400), (255, 255, 255), 2)

    # Displaying the main capture region separately
    cv2.imshow('data', frame)
    cv2.imshow('ROI', frame[40:400, 0:300])
    
    # Cropping the frame
    frame = frame[40:400, 0:300]
    
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == ord('a'):
        cv2.imwrite(directory + 'A/' + str(count['a']) + '.png', frame)
        
    if interrupt & 0xFF == ord('b'):
        cv2.imwrite(directory + 'B/' + str(count['b']) + '.png', frame)
            
    if interrupt & 0xFF == ord('c'):
        cv2.imwrite(directory + 'C/' + str(count['c']) + '.png', frame)
        

# Releasing the video capture device
cap.release()
cv2.destroyAllWindows()