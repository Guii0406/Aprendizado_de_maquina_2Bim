import numpy as np
import cv2
from matplotlib import pyplot as plt

face_cascade=cv2.CascadeClassifier('mydogdetector.xml')
face_cascade2=cv2.CascadeClassifier('mary3.xml')

img=cv2.imread('FotoMaryNaoTreinada.bmp')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
font=cv2.FONT_HERSHEY_SIMPLEX
faces=face_cascade.detectMultiScale(gray,1.345,5,75)
faces2=face_cascade2.detectMultiScale(gray,1.345,5,75)

for(x,y,w,h) in faces:
	img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
	cv2.putText(img,'Cachorro',(x,y),font,0.9,(0,255,0),2)

for(x,y,w,h) in faces2:
	img=cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
	cv2.putText(img,'Mary',(x,y),font,0.9,(255,0,0),2)

p,l,m=cv2.split(img)
img=cv2.merge([m,l,p])

plt.imshow(img)
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()

