import numpy as np
import math as m
import pandas as pd
from matplotlib import pyplot as plt
#Feito por João Gabriel, baseado no modelo apresentado por Samson Zhang no vídeo https://www.youtube.com/watch?v=w8yWXqWQYmU
#Base de dados: MNIST, disponível em https://www.kaggle.com/c/digit-recognizer/data
#Este código implementa uma rede neural simples com uma camada oculta usando apenas NumPy.
# O objetivo é treinar a rede para classificar dígitos manuscritos do dataset MNIST.

#leitura e organização do treino da rede
data = pd.read_csv('train.csv')
data = np.array(data)
m, n = data.shape #número de linhas e colunas
np.random.shuffle(data) #embaralha os dados

data_dev = data[0:1000].T #transpõe os dados das 1000 primeiras linhas
Y_dev = data_dev[0] #primeira linha é o rótulo
X_dev = data_dev[1:n] #restante das linhas são as features
X_dev = X_dev / 255.0 #normaliza os dados

data_train = data[1000:m].T #transpõe os dados do restante
Y_train = data_train[0]  
X_train = data_train[1:n] 
X_train = X_train / 255.0 
_,m_train = X_train.shape 

#inicialização dos parâmetros
def init_params():
    #normalização sugerida para uma performance de 0.90 de precisao
    W1 = np.random.normal(size=(10, 784)) * np.sqrt(1./(784))
    b1 = np.random.normal(size=(10, 1)) * np.sqrt(1./10)
    W2 = np.random.normal(size=(10, 10)) * np.sqrt(1./20)
    b2 = np.random.normal(size=(10, 1)) * np.sqrt(1./(784))
    return W1, b1, W2, b2
    
def ReLU(Z): #função de ativação ReLU
    return np.maximum(0, Z)

def softmax(Z): #função de calculo de probabilidade softmax
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    
def forward_prop(W1, b1, W2, b2, X): #propagação para frente
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, A2, Z2

def ReLU_deriv(Z): #derivada da função ReLU para backpropagation
    return Z > 0

def one_hot(Y): #converte os rótulos em formato one-hot
    one_hot_Y = np.zeros((Y.size, Y.max() + 1)) # cria matriz de zeros
    one_hot_Y[np.arange(Y.size), Y] = 1 # define os índices correspondentes aos rótulos como 1
    one_hot_Y = one_hot_Y.T # transpõe a matriz
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y): #backpropagation para calcular os gradientes
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y # calcula o erro na camada de saída
    dW2 = 1 / m * dZ2.dot(A1.T) # gradiente da camada de saída
    db2 = 1 / m * np.sum(dZ2)   # gradiente do bias da camada de saída
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1) # calcula o erro na camada oculta
    dW1 = 1 / m * dZ1.dot(X.T) # gradiente da camada oculta
    db1 = 1 / m * np.sum(dZ1) # gradiente do bias da camada oculta
    return dW1, db1, dW2, db2  

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha): #atualiza os parâmetros da rede
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2

def get_predictions(A2): #obtém as previsões da rede
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y): #calcula a acurácia das previsões
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations): #função principal de treinamento da rede
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X) 
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y) 
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha) 
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2

W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500) 

def make_predictions(X, W1, b1, W2, b2): #função para fazer previsões com os parâmetros treinados
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2): #função para testar uma previsão específica
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255 # redimensiona a imagem para 28x28
    plt.gray() # define o colormap como cinza
    plt.imshow(current_image, interpolation='nearest') # exibe a imagem
    plt.show()

#testa 4 instancias aleatorias (4 PRIMEIRAS DO ARRAY) a proposito de ilustrar a atividade da rede
test_prediction(0, W1, b1, W2, b2)
test_prediction(1, W1, b1, W2, b2)
test_prediction(2, W1, b1, W2, b2)
test_prediction(3, W1, b1, W2, b2)

dev_predictions = make_predictions(X_dev, W1, b1, W2, b2) # faz previsões para o conjunto de desenvolvimento
print(get_accuracy(dev_predictions, Y_dev)) # calcula e mostra a acurácia das previsões no conjunto de desenvolvimento

