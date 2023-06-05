import numpy as np
import cv2
import os

# Diretório onde as imagens de treinamento estão armazenadas
diretorio_treinamento = 'D:\\Importante\\2023 - 2\\Trabalho IA - 2\\treino_gudan'

# Inicializa o detector de faces
detector_faces = cv2.CascadeClassifier('D:\\Importante\\2023 - 2\\Trabalho IA - 2\\dog_face_haar_cascade-master\\cascade.xml')

# Função para extrair as características faciais de uma imagem
def extrair_caracteristicas(imagem):
    # Converte a imagem em escala de cinza
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    
    # Detecta as faces na imagem
    faces = detector_faces.detectMultiScale(cinza, scaleFactor=1.3, minNeighbors=5)
    
    # Retorna as regiões de interesse (faces) encontradas
    return faces

# Função para treinar o modelo de reconhecimento facial
def treinar_reconhecimento_facial(diretorio_treinamento):
    # Lista para armazenar as características faciais e os rótulos correspondentes
    caracteristicas = []
    rotulos = []
    
    # Loop sobre as imagens de treinamento
    for nome_arquivo in os.listdir(diretorio_treinamento):
        if nome_arquivo.endswith('D:\\Importante\\2023 - 2\\Trabalho IA - 2\\treino_gudan\\gudan 1') or nome_arquivo.endswith('.png'):
            # Lê a imagem
            imagem = cv2.imread(diretorio_treinamento + nome_arquivo)
            
            # Extrai as características faciais da imagem
            faces = extrair_caracteristicas(imagem)
            
            # Adiciona as características faciais e os rótulos à lista
            for (x, y, w, h) in faces:
                face = imagem[y:y+h, x:x+w]
                face = cv2.resize(face, (100, 100))  # Redimensiona a imagem para um tamanho fixo
                caracteristicas.append(face)
                rotulos.append(nome_arquivo)
    
    # Cria o modelo de reconhecimento facial
    reconhecimento = cv2.face.createLBPHFaceRecognizer()
    
    # Treina o modelo com as características faciais e os rótulos
    reconhecimento.train(caracteristicas, np.array(rotulos))
    
    # Retorna o modelo treinado
    return reconhecimento

# Função para reconhecer uma imagem usando o modelo treinado
def reconhecer_imagem(modelo, imagem):
    # Extrai as características faciais da imagem
    faces = extrair_caracteristicas(imagem)
    
    # Loop sobre as faces encontradas
    for (x, y, w, h) in faces:
        face = imagem[y:y + h, x:x + w]
        face = cv2.resize(face, (100, 100))  # Redimensiona a imagem para um tamanho fixo
        
        # Faz a previsão do rótulo para a face
        resultado, confianca = modelo.predict(face)
        
        # Verifica se o resultado corresponde a uma imagem de treinamento
        if confianca < 80:
            # Exibe o resultado
            cv2.putText(imagem, 'Cachorro: ' + resultado, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Retorna a imagem com as marcações
    return imagem

# Diretório da imagem que você deseja reconhecer
diretorio_imagem = 'D:\\Importante\\2023 - 2\\Trabalho IA - 2\\dog.jpg'

# Carrega o modelo treinado
modelo = treinar_reconhecimento_facial(diretorio_treinamento)

# Lê a imagem a ser reconhecida
imagem = cv2.imread(diretorio_imagem)

# Realiza o reconhecimento da imagem
imagem_reconhecida = reconhecer_imagem(modelo, imagem)

# Exibe a imagem reconhecida
cv2.imshow('Imagem Reconhecida', imagem_reconhecida)
cv2.waitKey(0)
cv2.destroyAllWindows()