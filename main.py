# entrada de dados onde cada neuronio vai receber um valor
# entrada vai ser um valor de matriz (n x m)
from PIL import Image
import numpy as np
import sys


def normalization(red, green, blue):
    gray_escale = (red * 0.299 + green * 0.587 + blue * 0.114).astype(np.uint8)
    neuron_value = gray_escale / 255
    return neuron_value


def hidden_layer(input, weight, bias):

    PLACEHOLDER_SUM = input @ weight  # soma total de ativação relacionando o array input e a matriz pesos (x_i * w_ij + bias)
    PLACEHOLDER_SUM += bias
   
    return PLACEHOLDER_SUM   

def sigmoid(ACTIVATION_VALUE):
    sigmoid_output = 1 / (1 + np.exp(-ACTIVATION_VALUE))
    return sigmoid_output

def relu(ACTIVATION_VALUE):
    RELU_OUTPUT = np.maximum(0, ACTIVATION_VALUE)
    return RELU_OUTPUT



def use_network():
    img = Image.open(r"C:\Users\pottv\Downloads\eletron\ai\image.jpg")
    RESHAPE = 64
    img_resized = img.resize((RESHAPE, RESHAPE)) # redimensiona a imagem para 64x64 pixels para não ficar gigante
    
    matriz = np.array(img_resized)
    CHANNELS = 3 # 3 canais de cores (RGB), se quiser mudar para hexa tem que lembrar de mudar aqui
    
    height, width, canais = matriz.shape
    neuron_matriz = np.zeros((height, width)) # matriz de neuronios que vai receber os valores normalizados

    #print("Matriz:", matriz)  # matriz 3d
    #print("Tipo:", type(matriz))  # tipo de array que a matriz será
    #print("Formato (h, w, c):", matriz.shape) # chape da matriz (height, width, channels) 
    #print("Pixel [0,0]:", matriz[0, 0]) # Matriz tridimensional, chamado de tensor por ser entrada do nosso sistema. Para uma imagem colorida, teremos 3 canais de cores (RGB) 
    
    # Processo de normalização; Acbamos de alocar um tensor de 3 dimensôes, mas temos que simplificar, podemos trabalhar com os valores separados ou 
    # transformar esses valores em escalas de cinza, ou seja, preto e branco.
    # formula de luminancia: 0.299*R + 0.587*G + 0.114*B
    

    for h in range(height):
        for w in range(width):
            neuron_value = normalization(matriz[h, w, 0], matriz[h, w, 1], matriz[h, w, 2])
            neuron_matriz[h, w] = neuron_value
            #print(neuron_matriz[h, w]) # imprimir valor do neuronio na posição 0,0

    #A função astype(np.uint8) é utilizada para converter valores de uma array de float para um tipo de dados unsigned 8 bits. 
    # Isso significa que os valores vão de 0 a 255
    
    # Tem que fazer a entrada ser enfileirada e ser transformada em uma matriz de 1 dimensão
    input = neuron_matriz.flatten() # flatten transforma a matriz em uma lista de valores
    

    # Primeira camada da rede neural
    # soma de todos os vallores . peso + bias é igual ao processo de ativação que o proximo neuronio vai receber

    input_count = RESHAPE * RESHAPE     
    neuron_count_layer = 32         
    

    # Trabalhando hoje


    weight = np.random.rand(input_count, neuron_count_layer) * 0.001
    bias = np.random.rand(neuron_count_layer)  # inicializa bias com valores aleatórios entre 0 e 1


    # Se quiser fazer na mão sem numpy, fui um pouco burrinho:
    #for i in range(input_count):
        #for j in range(neuron_count_layer):
            #weight[i][j] = np.random.rand()  # inicializa pesos com valores aleatórios entre 0 e 1   

    # valores serão aleatórios uma unica vez em toda rede
    
    ACTIVATION_VALUE = hidden_layer(input, weight, bias)
  
    use_sigmoid = False
    use_relu = False
    
    if sys.argv[1] == "sigmoid":
        use_sigmoid = True 

    if sys.argv[1] == "relu":
        use_relu = True

    if use_sigmoid == True:
        SIGMOID_OUTPUT = sigmoid(ACTIVATION_VALUE)
        print(SIGMOID_OUTPUT)

    if use_relu == True:
        RELU_OUTPUT = np.maximum(0, ACTIVATION_VALUE)
        print(RELU_OUTPUT)

use_network()

