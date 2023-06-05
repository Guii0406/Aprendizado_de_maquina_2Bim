import cv2
import numpy as np

def detectar_rosto_cachorro(imagem):
    # Carrega o classificador pré-treinado para detecção de faces de cachorros
    classificador_cachorro = cv2.CascadeClassifier('dog2.xml')
    
    # Converte a imagem em escala de cinza
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    
    # Detecta os rostos na imagem
    rostos_cachorros = classificador_cachorro.detectMultiScale(imagem_cinza,1.345,5,75)
    font=cv2.FONT_HERSHEY_SIMPLEX
    
    # Desenha um retângulo ao redor de cada rosto detectado
    for (x, y, w, h) in rostos_cachorros:
        cv2.rectangle(imagem, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(imagem,'Cachorro',(x,y),font,0.9,(0,255,0),2)
    
    # Retorna a imagem com os rostos detectados
    return imagem

# Carrega a imagem
imagem_cachorro = cv2.imread('cachorro.jpg')

# Realiza a detecção de rostos de cachorros
imagem_com_rostos = detectar_rosto_cachorro(imagem_cachorro)

# Exibe a imagem com os rostos detectados
cv2.imshow('Rostos de Cachorros', imagem_com_rostos)
cv2.waitKey(0)
cv2.destroyAllWindows()


