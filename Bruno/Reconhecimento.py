import numpy as np
import cv2
from matplotlib import pyplot as plt
  
# Carrega a imagem para identificar o cão
img = cv2.imread('D:\\Importante\\2023 - 2\\Trabalho IA - 2\\Mary\\Mary.bmp')
  
# Deixa a imagem cinza e realiza o resize 
# gray_img
face = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# face = cv2.resize(gray_img, (700, 600))
  
# Carrega o xml com o treinamento
haar_cascade = cv2.CascadeClassifier('D:\\Importante\\2023 - 2\\Trabalho IA - 2\\mydogdetector.xml')

haar_cascade2 = cv2.CascadeClassifier('D:\\Importante\\2023 - 2\\Trabalho IA - 2\\mary3.xml')
font=cv2.FONT_HERSHEY_SIMPLEX  

# Aplica o face detection
faces_rect = haar_cascade.detectMultiScale(face, 1.375, 5, 75)
faces_rect1 = haar_cascade2.detectMultiScale(face, 1.375, 5, 75)
  
# Marca o cão 
for (x, y, w, h) in faces_rect:
       cv2.rectangle(face, (x, y), (x+w, y+h), (0, 255, 0), 2)
       cv2.putText(face,'Cachorro',(x,y),font, 0.9, (0, 255, 0), 2)
        
for(x, y, w, h) in faces_rect1:
	face=cv2.rectangle(face,(x,y),(x+w, y+h), (255, 0, 0), 2)
	cv2.putText(face,'Mary',(x,y),font, 0.9, (255, 0, 0), 2)

plt.imshow(face)
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()