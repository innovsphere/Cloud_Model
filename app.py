import os
import cvzone
from cvzone.ClassificationModule import Classifier
import cv2

# Initialize the camera
cap = None
for i in range(3):  # Try camera indices 0, 1, 2
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Using camera index {i}")
        break
else:
    print("Error: No camera found!")
    exit()

# Load the classifier
classifier = Classifier('keras_model.h5', 'labels.txt')
imgArrow = cv2.imread('arrow.png', cv2.IMREAD_UNCHANGED)
classIDBin = 0

# Import all the waste images
imgWasteList = []
pathFolderWaste = "Waste"
pathList = os.listdir(pathFolderWaste)
for path in pathList:
    imgWasteList.append(cv2.imread(os.path.join(pathFolderWaste, path), cv2.IMREAD_UNCHANGED))

# Import all the bin images
imgBinsList = []
pathFolderBins = "Bins"
pathList = os.listdir(pathFolderBins)
for path in pathList:
    imgBinsList.append(cv2.imread(os.path.join(pathFolderBins, path), cv2.IMREAD_UNCHANGED))

# Classification dictionary
classDic = {0: None,
            1: 0,
            2: 0,
            3: 3,
            4: 3,
            5: 1,
            6: 1,
            7: 2,
            8: 2}

while True:
    # Capture frame from camera
    ret, img = cap.read()
    if not ret or img is None:
        print("Error: Failed to capture image!")
        continue

    imgResize = cv2.resize(img, (454, 340))
    imgBackground = cv2.imread('background.png')

    # Perform prediction
    prediction = classifier.getPrediction(img)
    classID = prediction[1]
    print(classID)

    if classID != 0:
        imgBackground = cvzone.overlayPNG(imgBackground, imgWasteList[classID - 1], (909, 127))
        imgBackground = cvzone.overlayPNG(imgBackground, imgArrow, (978, 320))
        classIDBin = classDic[classID]

    imgBackground = cvzone.overlayPNG(imgBackground, imgBinsList[classIDBin], (895, 374))
    imgBackground[148:148 + 340, 159:159 + 454] = imgResize

    # Display the output
    cv2.imshow("Output", imgBackground)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
