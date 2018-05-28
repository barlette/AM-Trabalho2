import sys
import numpy as np
import pandas as pd
import csv
import math

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

inputs = np.matrix([[1], [0.13]])
outputs = np.matrix([[0.9]])

print("Conjunto de treinamento:")
print("x:", inputs[1])
print("y:", outputs)
print("--------------------------------------------")
print("Calculando erro/custo J da rede")
print("Propagando entrada", inputs[1])
activations = []
activations.append(inputs)

for layer in range(len(weightLines)):
	activations.append(sigmoid(np.matmul(weightLines[layer], activations[layer])))
	activations[layer+1] = np.vstack([1,activations[layer+1]])
	print("a",layer,":",activations[layer].T)
	#print(sigmoid(activations[layer]))


#print(len(activations[len(activations)-1]))
#print(activations[len(activations)-1][1:len(activations[len(activations)-1])])
index = len(activations)-1
outputActivations = activations[index][1:len(activations[index])]
print("a",len(activations),":",outputActivations)
# totalError = np.sum(0.5*(np.power(outputs - outputActivations,2)))
# print("Total error: ",totalError)
#np.log(np.squeeze(np.asarray(outputActivations)))
print("Saida predita:", outputActivations)
print("Saida esperada:", outputs)
print("J:", -np.multiply(outputs, (np.log(np.squeeze(np.asarray(outputActivations))))) -np.multiply((1-outputs),  (np.log(1-np.squeeze(np.asarray(outputActivations))))))
print("--------------------------------------------")
print("Rodando backpropagation")
print("Calculando gradientes")
delta = outputActivations - outputs
print("delta", len(layers), ":", delta)

activationsPL = activations[index-1][1:len(activations[index-1])]
weightNoBias = weightLines[index-1][0][1:len(weightLines[index-1][0])]
print(weightNoBias)
print(activationsPL)
delta2 = np.multiply(np.multiply(weightNoBias,delta).sum(axis=0), np.multiply(activationsPL, (1-activationsPL)).T)
print("delta", len(layers)-1, ":", delta2)

print("Gradientes de Theta 2")
grad = np.multiply(activationsPL, delta).T 
print(grad)

print(weightLines[index-1])
attWeight = weightLines[index-1] -regFactor*delta
print(attWeight)

index = index-1
activationsPL = activations[index-1]
print("Gradientes de Theta 1")
grad2 = np.multiply(activationsPL, delta2).T
print(grad2)
attWeight2 = weightLines[index-1] -regFactor*delta2
print(attWeight2)
print("--------------------------------------------")

# slopeOut = np.zeros((len(outputActivations),1))
# slopeOut = derivativeSig(outputActivations)
# print("slopeOut:", slopeOut)

# delta2 = np.multiply(delta,slopeOut)
# print("delta:", delta2)
# #delta = np.ndarray.transpose(delta)
# #activationsPL = activations[index-1][1:len(activations[index-1])]
# activationsPL = activations[index-1]
# print("activationsPL:", activationsPL)
# print(delta.T)
# gradiente = np.matmul(activationsPL, delta.T)
# print("der:", gradiente)



# attWeight = weightLines[index-1] -(regFactor*gradiente).T
# print(attWeight)

# #back propagation das camadas internas

# index = index-1
# #print(activations[index])
# #dEtotal/douth1 = dEout1/douth1 + dEout2/douth1
# #dEout1/dnetout1 = deltaout1
# print("inside backpropagation")
# print(weightLines[index])
# print(delta)
# dInside = np.multiply(weightLines[index], delta)
# print(dInside)
# sums = dInside.sum(axis=0)
# print("sums:", sums)
# activationsInside = activations[index][1:len(activations[index])]
# print(activationsInside)
# douth1 = derivativeSig(activationsInside)
# print("douth:", douth1)
# print(activations[index-1])
# print(np.matmul(activations[index-1], douth1.T))
# dEtotal = np.matmul(sums, np.matmul(activations[index-1], douth1.T))
# print(dEtotal)
# attInternalWeight = weightLines[index-1] -(regFactor*dEtotal).T
# print(attInternalWeight)
# #attWeight.append()