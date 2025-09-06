# entrada de dados onde cada neuronio vai receber um valor
# entrada vai ser um valor de matriz (n x m)
from PIL import Image
import numpy as np

def input_data_network(inputs, weights, bias):
    img = Image.open(r"C:\Users\pottv\Downloads\eletron\ai\image.jpg")
    matriz = np.array(img)

    print("Tipo:", type(matriz))  # tipo de array que a matriz será
    print("Formato (h, w, c):", matriz.shape) # chape da matriz (height, width, channels) 
    print("Pixel [0,0]:", matriz[0, 0]) # Matriz tridimensional, chamado de tensor por ser entrada do nosso sistema. Para uma imagem colorida, teremos 3 canais de cores (RGB) 
    
    # Processo de normalização; Acbamos de alocar um tensor de 3 dimensôes, mas temos que simplificar, podemos trabalhar com os valores separados ou 
    # transformar esses valores em escalas de cinza, ou seja, preto e branco.
    # formula de luminancia: 0.299*R + 0.587*G + 0.114*B
    
    
    matriz_normalizada = (matriz[0,0,0] * 0.299 + matriz[0,0,1] * 0.587 + matriz[0,0,2] * 0.114).astype(np.uint8)
    cinza_normalizado = matriz_normalizada / 255
    #A função astype(np.uint8) é utilizada para converter valores de uma array de float para um tipo de dados unsigned 8 bits. 
    # Isso significa que os valores vão de 0 a 255
    # Tem que fazer a entrada ser enfileirada e ser transformada em uma matriz de 1 dimensão
    input = cinza_normalizado.flatten()
    print("Matriz normalizada:", matriz_normalizada) 
    
input_data_network(0, 0, 0)
# soma de todos os vallores . peso + bias é igual ao processo de ativação que o proximo neuronio vai receber

