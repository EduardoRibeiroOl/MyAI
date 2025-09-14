# entrada de dados onde cada neuronio vai receber um valor
# entrada vai ser um valor de matriz (n x m)
from PIL import Image
import numpy as np


def normalization(red, green, blue):
    gray_escale = (red * 0.299 + green * 0.587 + blue * 0.114).astype(np.uint8)
    neuron_value = gray_escale / 255
    return neuron_value

def input_data_network():
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

    input_count = RESHAPE * RESHAPE       # quantidade de entradas (ex: 64*64 = 4096)
    neuron_count_layer = 32               # quantidade de neurônios da camada

    PLACEHOLDER_SUM = []   # lista que vai armazenar a saída de cada neurônio da camada
    PLACEHOLDER_BIAS = []  # lista que vai armazenar o bias de cada neurônio

    # Loop sobre os neurônios (um de cada vez)
    for j in range(neuron_count_layer):

        soma = 0  # acumulador da soma para o neurônio j

        # Loop sobre todas as entradas
        for i in range(input_count):
            weight = np.random.rand()   # gera um peso aleatório para a ligação i->j
            soma += input[i] * weight   # soma parcial (x_i * w_ij)

        # Agora que percorri todos os inputs, adiciono o bias do neurônio j
        bias = np.random.rand()          # bias único para o neurônio j
        soma += bias

        # Guardo os valores
        PLACEHOLDER_SUM.append(soma)     # saída bruta do neurônio (z_j)
        PLACEHOLDER_BIAS.append(bias)    # bias usado neste neurônio

        print(f"Neurônio {j}: Saída (z_j) = {soma:.4f}, Bias = {bias:.4f}")

    #implementação de sigmoid







input_data_network()

