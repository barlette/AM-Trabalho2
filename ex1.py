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

for layer in range(len(layers)):
    layers[layer] = float(layers[layer])

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
print(layers[0]-1, "input neurons")
print("Network:")

for layer in range(len(weightLines)):
	print("Layer", layer+1)
	for row in range(len(weightLines[layer])):
		for neuron in range(len(weightLines[layer][row])):
			print("Neuron", neuron+1, ", Weight:", weightLines[layer][row][neuron])

#forward propagation

#print(norm[1:2])

inputs = np.matrix([[1], [0.05], [0.1]])
outputs = np.matrix([[0.01], [0.99]])

activations = []
activations.append(inputs)

#print(activations[0])
#print(weightLines[0])
#print(np.matmul(weightLines[0], activations[0]))

for layer in range(len(weightLines)):
	# print(weightLines[layer])
	# print(activations[layer])
	# print("teste1")
	# print(np.matmul(weightLines[layer], activations[layer]))
	activations.append(sigmoid(np.matmul(weightLines[layer], activations[layer])))
	activations[layer+1] = np.vstack([1,activations[layer+1]])

print(activations)

#print(len(activations[len(activations)-1]))
#print(activations[len(activations)-1][1:len(activations[len(activations)-1])])
index = len(activations)-1
outputActivations = activations[index][1:len(activations[index])]

#totalError = np.sum(0.5*(np.power(outputs - activations[len(activations)-1],2)))
#print("Total error: ",totalError)

print(activations[index])
error = outputActivations - outputs
print("error:", error)

slopeOut = np.zeros((len(outputActivations),1))
slopeOut = derivativeSig(outputActivations)
print("slopeOut:", slopeOut)

delta = np.multiply(error,slopeOut)
print("delta:", delta)
#delta = np.ndarray.transpose(delta)
#activationsPL = activations[index-1][1:len(activations[index-1])]
activationsPL = activations[index-1]
print(activationsPL)
print(delta.T)
gradiente = np.matmul(activationsPL, delta.T)
print("der:", gradiente)

print(weightLines[index-1])

attWeight = weightLines[index-1] -(regFactor*gradiente).T
print(attWeight)

