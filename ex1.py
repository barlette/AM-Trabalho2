import sys
import numpy
import pandas
import csv

network = open(sys.argv[1], "r")
weights = open(sys.argv[2], "r")

data = pandas.read_csv(sys.argv[3], sep = ',', header = 0)
print(data)

layers = network.readlines()

regFactor = float(layers[0])

layers.pop(0)

for layer in range(len(layers)):
    layers[layer] = float(layers[layer])
    
print(layers)

weightLines = weights.readlines()

wMatrix = numpy.zeros((0,0))

for layer in range(len(weightLines)):
	weightLines[layer] = weightLines[layer].split(';')
	print(len(weightLines[layer]))
	for inputs in range(len(weightLines[layer])):
		weightLines[layer][inputs] = weightLines[layer][inputs].split(',')		
		for nW in range(len(weightLines[layer][inputs])):
			weightLines[layer][inputs][nW] = float(weightLines[layer][inputs][nW])

print(weightLines)

