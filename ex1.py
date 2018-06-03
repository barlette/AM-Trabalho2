import sys
import numpy as np
import pandas as pd
import csv
import math
import time
import random

def divideIntoKFolds(kFolds, fileName):
    #Variables
    Classifications = []
    rawdata = []
    dataset = []

    #Vamos abrir o arquivo de dados e armazenar em formato string
    # print('Opening', fileName, '...')
    with open(fileName, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        rawdata = list(spamreader)
    # print('Dataset successfully read from', fileName)

    #Vamos remover o header do dataframe
    # print('Processing dataset ...')
    del rawdata[0]
    #Vamos converter a lista de strings em lista de valores float
    for item in rawdata:
        lista1 = item[0].split(',')
        flist = [float(i) for i in lista1]
        Classifications.append(int(flist.pop(0)))
        dataset.append(flist)

    num_of_classes = len(set(Classifications))
    # print('Dataset overview:')
    # for nClass in range(1,num_of_classes+1):
    #     print('Class', nClass, '= ', Classifications.count(nClass), 'instances')

    #Vamos percorrer a lista de dados e separar em n pilhas de acordo com a classe
    dct = {}
    instances_list = []
    for nClass in range(1,num_of_classes+1):
        for nIndex in range(0, len(dataset)):
            if Classifications[nIndex] == nClass:
                instances_list.append(dataset[nIndex])
        dct['%s' % nClass] = instances_list
        instances_list = []

    # print('Creating kFolds...')

    # print(random.randint(1,4))
    dataFolds = []
    tempList = []
    randomIndex = 0
    for nClass in range(1,num_of_classes+1):
        instances_list = dct.get('%s' %nClass)
        while len(instances_list) > 0 :
            if len(dataFolds) < kFolds:
                randomIndex = random.randint(0,len(instances_list)-1)
                tempList.append(instances_list.pop(randomIndex))
                dataFolds.append(tempList)
                tempList = []
            else:
                for fold in dataFolds:
                    if not instances_list:
                        break
                    randomIndex = random.randint(0,len(instances_list)-1)
                    fold.append(instances_list.pop(randomIndex))

    # for fold in dataFolds:
    #     print('Fold[',dataFolds.index(fold),'] created with',len(fold),'instances')

    # print('\n\nPress \'Enter\' to Finish...')
    # input()
    return dataFolds

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def normalize(df):
	return (df - df.min()) / (df.max() - df.min())

def derivativeSig(x):
	return np.multiply(x, (1-x))

#parsing dos arquivos

network = open(sys.argv[1], "r")
weights = open(sys.argv[2], "r")


data = pd.read_csv(sys.argv[3], sep = ',', header = 0)

#print(data[1:3])
norm = normalize(data)
#print(norm)

layers = network.readlines()

regFactor = float(layers[0])

layers.pop(0)
corLayers = np.zeros((len(layers),1))
#print(len(layers))
#print(len(corLayers))
for layer in range(len(layers)):
    layers[layer] = float(layers[layer])
    #print(layers[layer]-1)
    corLayers[layer] = layers[layer]-1

corLayers[len(corLayers)-1] = corLayers[len(corLayers)-1]+1

weightLines = weights.readlines()

wMatrix = np.zeros((0,0))

for layer in range(len(weightLines)):
	weightLines[layer] = weightLines[layer].split(';')
	#print(len(weightLines[layer]))
	for inputs in range(len(weightLines[layer])):
		weightLines[layer][inputs] = weightLines[layer][inputs].split(',')		
		for nW in range(len(weightLines[layer][inputs])):
			weightLines[layer][inputs][nW] = float(weightLines[layer][inputs][nW])

#impressao das informações
#print(layers[0]-1, "input neurons")
#print("Network:")
print("Parametro de regularizacao lambda: ", regFactor)
print("Inicializando rede com a seguinte estrutura de neuronios por camadas:", corLayers.T)

for layer in range(len(weightLines)):
	print("Theta",layer+1,"inicial (pesos de cada neuronio, incluindo bias, armazenados nas linhas):")
	for row in range(len(weightLines[layer])):
		print(weightLines[layer][row])
# 		for neuron in range(len(weightLines[layer][row])):
# 			print("Neuron", neuron+1, ", Weight:", weightLines[layer][row][neuron])

#forward propagation

#print(norm[1:2])

inputs = np.matrix([[1], [0.32], [0.68]])
outputs = np.matrix([[0.75], [0.98]])

print("Conjunto de treinamento:")
print("x:", inputs[1])
print("y:", outputs)
print("--------------------------------------------")
print("Calculando erro/custo J da rede")
print("Propagando entrada", inputs[1])
activations = []
activations.append(inputs)
z = []
z.append(inputs)

for layer in range(len(weightLines)):
	activations.append(sigmoid(np.matmul(weightLines[layer], activations[layer])))
	z.append(np.matmul(weightLines[layer], activations[layer]))
	activations[layer+1] = np.vstack([1,activations[layer+1]])
	z[layer+1] = np.vstack([1,z[layer+1]])
	print("z",layer,":",z[layer].T)
	print("a",layer,":",activations[layer].T)

index = len(activations)-1
outputActivations = activations[index][1:len(activations[index])]
outputZ = z[index][1:len(z[index])]
print("z",len(z),":",outputZ.T)
print("a",len(activations),":",outputActivations.T)

print("Saida predita:", np.squeeze(np.asarray(outputActivations)))
print("Saida esperada:", np.squeeze(np.asarray(outputs)))
print("J:", (-np.multiply(np.squeeze(np.asarray(outputs)), (np.log(np.squeeze(np.asarray(outputActivations))))) -np.multiply((1-np.squeeze(np.asarray(outputs))),  (np.log(1-np.squeeze(np.asarray(outputActivations)))))).sum(axis=0))
print("--------------------------------------------")
print("Rodando backpropagation")
print("Calculando gradientes")

activationsPL = activations[index-1]
weightNoBias = weightLines[index-1]

deltaNoBias = outputActivations - outputs
print("delta", len(layers), ":", deltaNoBias)

print("Gradientes de Theta", index)
grad = np.multiply(activationsPL, deltaNoBias.T).T
print(grad)

while index > 1:
	print(activationsPL)
	print(weightNoBias)
	delta = np.multiply(np.multiply(weightNoBias,deltaNoBias).sum(axis=0), np.multiply(activationsPL, (1-activationsPL)).T)

	deltaNoBias = delta[0,1:len(np.squeeze(np.asarray(delta)))]
	print("delta", len(layers)-1, ":", deltaNoBias)
	temp = delta[0,]

	index = index-1
	activationsPL = activations[index-1]
	weightNoBias = weightLines[index-1]
	print("Gradientes de Theta", index)
	grad = np.multiply(activationsPL, deltaNoBias).T
	print(grad)
	deltaNoBias = deltaNoBias.T

print("--------------------------------------------")
