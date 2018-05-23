import sys
import numpy as np
import pandas as pd
import csv
import math

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def normalize(df):
	return (df - df.min()) / (df.max() - df.min())

#parsing dos arquivos

network = open(sys.argv[1], "r")
weights = open(sys.argv[2], "r")


data = pd.read_csv(sys.argv[3], sep = ',', header = 0)

#print(data[1:3])
norm = normalize(data)
print(norm)

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

inputs = np.matrix([[1], [2]])

activations = []
activations.append(inputs)

#print(activations[0])
#print(weightLines[0])
#print(np.matmul(weightLines[0], activations[0]))

for layer in range(len(weightLines)):
	activations.append(sigmoid(np.matmul(weightLines[layer], activations[layer])))
	activations[layer+1] = np.vstack([1,activations[layer+1]])

print(activations)
