import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

cap = cv2.VideoCapture(0)#here "0"means builtin cam is used we can give 1/2/3 also for various cams there

while 1:

    #reads the every frames in the cam
    ret, img = cap.read()

    #coverts the grayscale of each frame because th classifier is trained for grayscale detection only
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #here we are giving scalefactor and min neighbour
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # loop runs if capturing has been initialized. 
#while 1:  
  
    # reads frames from a camera 
    #ret, img = cap.read()  
  
    # convert to gray scale of each frames 
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
  
    # Detects faces of different sizes in the input image 
   #faces = face_cascade.detectMultiScale(gray, 1.3, 5) 

    for (x,y,w,h) in faces:
      cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
      roi_gray = gray[y:y+h, x:x+w]
      rai_color = img[y:y+h, x:x+w]

    eyes = eye_cascade.detectMultiScale(roi_gray)

    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(rai_color,(ex,ey),(ex+ew,ey+eh),(0,127,255),2)
    # Display an image in a window

    cv2.imshow("img",img)
    # Wait for Esc key to stop 
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()    

cv2.destroyAllWindows()        
