import sys
import numpy as np
import pandas as pd
import csv
import math
import time
import random
import copy

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
        Classifications.append(int(flist[0]))
        dataset.append(flist)

    num_of_classes = len(set(Classifications))

    norm = np.zeros(shape=(np.size(dataset,0),np.size(dataset,1)))
    for column in range(np.size(dataset,1)):
    	norm[:,column] = normalize(np.asmatrix(dataset)[0:, column]).T

    #Vamos percorrer a lista de dados e separar em n pilhas de acordo com a classe
    dct = {}
    instances_list = []
    for nClass in range(1,num_of_classes+1):
        for nIndex in range(0, len(norm)):
            if Classifications[nIndex] == nClass:
                instances_list.append(norm[nIndex])
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

def forwardProp(inputs, outputs, weightLines, regFactor):
	regJ = []
	regJSum = 0
	for(layer) in range(len(weightLines)):
		regJ.append(np.power(weightLines[layer],2))
		regJSum = regJSum + np.sum(regJ[layer][0:len(regJ[layer]),1:])
		
	regJSum = (regFactor/(2*len(inputs)))*regJSum
	J = []
	activations = []
	z = []
	outputActivations = []
	for inputxDex in range(len(inputs)):
		# print("Conjunto de treinamento:")
		# print("x:", np.asmatrix(inputs[inputxDex]).T)
		# print("y:", outputs[inputxDex])
		# print("--------------------------------------------")
		# print("Calculando erro/custo J da rede")
		# print("Propagando entrada", np.asmatrix(inputs[inputxDex]).T)
		activations.append([])
		activations[inputxDex].append(inputs[inputxDex].T)
		z.append([])
		z[inputxDex].append(inputs[inputxDex].T)
		for layer in range(len(weightLines)):
			# print(weightLines[layer])
			# print(activations[inputxDex][layer])
			# print(np.matmul(weightLines[layer], activations[inputxDex][layer]))
			activations[inputxDex].append(sigmoid(np.matmul(weightLines[layer], activations[inputxDex][layer])))
			z[inputxDex].append(np.matmul(weightLines[layer], activations[inputxDex][layer]))
			activations[inputxDex][layer+1] = np.vstack([1,activations[inputxDex][layer+1]])
			z[inputxDex][layer+1] = np.vstack([1,z[inputxDex][layer+1]])
			# print("z",layer,":",z[inputxDex][layer].T)
			# print("a",layer,":",activations[inputxDex][layer].T)
		index = len(activations[inputxDex])-1
		outputActivations.append(activations[inputxDex][index][1:len(activations[inputxDex][index])])
		outputZ = z[inputxDex][index][1:len(z[inputxDex][index])]
		# print("z",len(z[inputxDex]),":",outputZ.T)
		# print("a",len(activations[inputxDex]),":",outputActivations[inputxDex].T)
		# print("Saida predita:", np.squeeze(np.asarray(outputActivations[inputxDex])))
		# print("Saida esperada:", np.squeeze(np.asarray(outputs[inputxDex])))
		J.append((-np.multiply(np.squeeze(np.asarray(outputs[inputxDex])), (np.log(np.squeeze(np.asarray(outputActivations[inputxDex]))))) -np.multiply((1-np.squeeze(np.asarray(outputs[inputxDex]))),  (np.log(1-np.squeeze(np.asarray(outputActivations[inputxDex])))))).sum(axis=0))
		print("J:", J[inputxDex])
	print("J final:", (np.sum(J)/len(J))+regJSum)


network = open(sys.argv[1], "r")
weights = open(sys.argv[2], "r")
layers = network.readlines()
regFactor = float(layers[0])

layers.pop(0)
corLayers = np.zeros((len(layers),1))

for layer in range(len(layers)):
    layers[layer] = float(layers[layer])
    corLayers[layer] = layers[layer]-1

corLayers[len(corLayers)-1] = corLayers[len(corLayers)-1]+1
initialWeightLines = weights.readlines()
wMatrix = np.zeros((0,0))

for layer in range(len(initialWeightLines)):
	initialWeightLines[layer] = initialWeightLines[layer].split(';')
	for inputs in range(len(initialWeightLines[layer])):
		initialWeightLines[layer][inputs] = initialWeightLines[layer][inputs].split(',')		
		for nW in range(len(initialWeightLines[layer][inputs])):
			initialWeightLines[layer][inputs][nW] = float(initialWeightLines[layer][inputs][nW])

#print("Parametro de regularizacao lambda: ", regFactor)
#print("Inicializando rede com a seguinte estrutura de neuronios por camadas:", corLayers.T)

for layer in range(len(initialWeightLines)):
	#print("Theta",layer+1,"inicial (pesos de cada neuronio, incluindo bias, armazenados nas linhas):")
	for row in range(len(initialWeightLines[layer])):
		print(initialWeightLines[layer][row])

folds = divideIntoKFolds(10, sys.argv[3])
for nFold in range(len(folds)):
	print("Fold:", nFold)
	weightLines = initialWeightLines
	#fold de teste
	testing = copy.deepcopy(folds[nFold])
	#o resto para treinamento
	inputsTotal = np.delete(folds, nFold)
	inputsTotal = np.concatenate(inputsTotal[0:])
	#as saídas esperadas
	outputs = copy.deepcopy(np.asmatrix(inputsTotal[0:, 0]).T)
	#print("outputs:", outputs)
	#inserindo o bias
	inputsTotal[0:, 0] = np.ones(len(inputsTotal))

	minibatch = 1
	batchIndex = 0
	while minibatch==1:
		if(((batchIndex+1)*50) < len(inputsTotal)):
			inputs = inputsTotal[batchIndex*50:((batchIndex+1)*50)-1,:]
		else:
			inputs = inputsTotal[batchIndex*50:len(inputsTotal),:]
			minibatch = 0
		batchIndex = batchIndex+1
		regJ = []
		regJSum = 0
		for(layer) in range(len(weightLines)):
			for row in range(len(weightLines[layer])):
				#print(row)
				if row==0:
					regJ.append(np.power(np.array(weightLines[layer][row]),2))
				else:
					regJ[layer] = np.vstack((regJ[layer], np.power(np.array(weightLines[layer][row]),2)))
				#print(regJ[layer])
			#print(regJ[layer][0:][1:])
			regJSum = regJSum + np.sum(regJ[layer][0:][1:])
		
		regJSum = (regFactor/(2*len(inputs)))*regJSum
		J = []
		activations = []
		z = []
		outputActivations = []
		for inputxDex in range(len(inputs)):
			#print("Conjunto de treinamento:")
			#print("x:", np.asmatrix(inputs[inputxDex]).T)
			#print("y:", outputs[inputxDex])
			#print("--------------------------------------------")
			#print("Calculando erro/custo J da rede")
			#print("Propagando entrada", np.asmatrix(inputs[inputxDex]).T)
			activations.append([])
			activations[inputxDex].append(np.asmatrix(inputs[inputxDex]).T)
			z.append([])
			z[inputxDex].append(np.asmatrix(inputs[inputxDex]).T)
			for layer in range(len(weightLines)):
				#print(weightLines[layer])
				#print(activations[inputxDex][layer])
				#print(np.matmul(weightLines[layer], activations[inputxDex][layer]))
				activations[inputxDex].append(sigmoid(np.matmul(weightLines[layer], activations[inputxDex][layer])))
				z[inputxDex].append(np.matmul(weightLines[layer], activations[inputxDex][layer]))
				activations[inputxDex][layer+1] = np.vstack([1,activations[inputxDex][layer+1]])
				z[inputxDex][layer+1] = np.vstack([1,z[inputxDex][layer+1]])
				#print("z",layer,":",z[inputxDex][layer].T)
				#print("a",layer,":",activations[inputxDex][layer].T)
			index = len(activations[inputxDex])-1
			outputActivations.append(activations[inputxDex][index][1:len(activations[inputxDex][index])])
			outputZ = z[inputxDex][index][1:len(z[inputxDex][index])]
			#print("z",len(z[inputxDex]),":",outputZ.T)
			#print("a",len(activations[inputxDex]),":",outputActivations[inputxDex].T)
			#print("Saida predita:", np.squeeze(np.asarray(outputActivations[inputxDex])))
			#print("Saida esperada:", np.squeeze(np.asarray(outputs[inputxDex])))
			J.append((-np.multiply(np.squeeze(np.asarray(outputs[inputxDex])), (np.log(np.squeeze(np.asarray(outputActivations[inputxDex]))))) -np.multiply((1-np.squeeze(np.asarray(outputs[inputxDex]))),  (np.log(1-np.squeeze(np.asarray(outputActivations[inputxDex])))))).sum(axis=0))
			#print("J:", J[inputxDex])
		print("\nJ total do dataset (com regularizacao):", (np.sum(J)/len(J))+regJSum)
		print("\n\n-------------------------------------------")

		grad = []
		for inputxDex in range(len(inputs)):
			gradIndex = 0
			#print(len(activations[inputxDex]))
			index = len(activations[inputxDex])-1
			#print("Rodando backpropagation")
			#print("Calculando gradientes")
			activationsPL = activations[inputxDex][index-1]
			weightNoBias = weightLines[index-1]
			deltaNoBias = outputActivations[inputxDex] - outputs[inputxDex]
			#print("delta", len(layers), ":", deltaNoBias.T)
			#print("Gradientes de Theta", index, ", input:", inputxDex)
			gradtmp = np.multiply(activationsPL, deltaNoBias.T).T
			if inputxDex == 0:
				grad.append(gradtmp)
			else:
				grad[gradIndex] = grad[gradIndex] + gradtmp
			#print(gradtmp)

			while (index > 1):
				gradIndex = gradIndex+1
				delta = np.multiply(np.multiply(weightNoBias,deltaNoBias).sum(axis=0), np.multiply(activationsPL, (1-activationsPL)).T)
				deltaNoBias = delta[0,1:len(np.squeeze(np.asarray(delta)))]
				#print("delta", index, ":", deltaNoBias)
				temp = delta[0,]
				index = index-1
				activationsPL = activations[inputxDex][index-1]
				weightNoBias = weightLines[index-1]
				#print("Gradientes de Theta", index, ", input:", inputxDex)
				gradtmp = np.multiply(activationsPL, deltaNoBias).T
				if inputxDex == 0:
					grad.append(gradtmp)
				else:
					grad[gradIndex] = grad[gradIndex] + gradtmp
				#print(gradtmp)
				deltaNoBias = deltaNoBias.T

		weightsNoBias = []
		for wLayer in range(len(weightLines)):
			teste = np.asmatrix(weightLines[wLayer])
			teste[:,0] = 0
			weightsNoBias.append(regFactor*teste)
		grad.reverse()
		grad = np.add(grad, weightsNoBias)
		grad = 1/len(inputs)*grad
		#print("Dataset completo processado. Calculando gradientes regularizados")
		#print("Pesos antigos:", weightLines)
		#for wLayer in range(len(grad)):
		 	#print(np.asmatrix(weightLines[wLayer]))
		 	#print("Gradientes finais para Theta", wLayer+1, "(com regularizacao):")
		 	#print(grad[wLayer])
		#print("\n\n--------------------------------------------")
		weightLines = weightLines- regFactor*(grad)
		#print("Pesos atualizados:", weightLines)
		
		#print("testing:", testing)
		#print("testing:", testing[0:][0])
	inputTesting = np.asmatrix(testing)
	outputsTesting = copy.deepcopy(np.asmatrix(testing)[0:, 0])
		#print("outputsTesting:", outputsTesting)
	inputTesting[:, 0] = 1
		#print("input testing:", inputTesting)
	forwardProp(inputTesting, outputsTesting, weightLines, regFactor)


# =============================================================

# inputs = []
# inputs.append(np.matrix([[1], [0.13]]))
# inputs.append(np.matrix([[1], [0.42]]))
# inputs.append(np.matrix([[1], [0.32], [0.68]]))
# inputs.append(np.matrix([[1], [0.83], [0.02]]))
# outputs = []
# outputs.append(np.matrix([[0.9]]))
# outputs.append(np.matrix([[0.23]]))
# outputs.append(np.matrix([[0.75], [0.98]]))
# outputs.append(np.matrix([[0.75], [0.28]]))

# =============================================================

# regJ = []
# regJSum = 0
# for(layer) in range(len(weightLines)):
# 	regJ.append(np.power(weightLines[layer],2))
# 	regJSum = regJSum + np.sum(regJ[layer][0:len(regJ[layer]),1:])
	
# regJSum = (regFactor/(2*len(inputs)))*regJSum
# J = []
# activations = []
# z = []
# outputActivations = []
# for inputxDex in range(len(inputs)):
# 	print("Conjunto de treinamento:")
# 	print("x:", inputs[inputxDex])
# 	print("y:", outputs[inputxDex])
# 	print("--------------------------------------------")
# 	print("Calculando erro/custo J da rede")
# 	print("Propagando entrada", inputs[inputxDex])
# 	activations.append([])
# 	activations[inputxDex].append(inputs[inputxDex])
# 	z.append([])
# 	z[inputxDex].append(inputs[inputxDex])
# 	for layer in range(len(weightLines)):
# 		activations[inputxDex].append(sigmoid(np.matmul(weightLines[layer], activations[inputxDex][layer])))
# 		z[inputxDex].append(np.matmul(weightLines[layer], activations[inputxDex][layer]))
# 		activations[inputxDex][layer+1] = np.vstack([1,activations[inputxDex][layer+1]])
# 		z[inputxDex][layer+1] = np.vstack([1,z[inputxDex][layer+1]])
# 		print("z",layer,":",z[inputxDex][layer].T)
# 		print("a",layer,":",activations[inputxDex][layer].T)
# 	index = len(activations[inputxDex])-1
# 	outputActivations.append(activations[inputxDex][index][1:len(activations[inputxDex][index])])
# 	outputZ = z[inputxDex][index][1:len(z[inputxDex][index])]
# 	print("z",len(z[inputxDex]),":",outputZ.T)
# 	print("a",len(activations[inputxDex]),":",outputActivations[inputxDex].T)
# 	print("Saida predita:", np.squeeze(np.asarray(outputActivations[inputxDex])))
# 	print("Saida esperada:", np.squeeze(np.asarray(outputs[inputxDex])))
# 	J.append((-np.multiply(np.squeeze(np.asarray(outputs[inputxDex])), (np.log(np.squeeze(np.asarray(outputActivations[inputxDex]))))) -np.multiply((1-np.squeeze(np.asarray(outputs[inputxDex]))),  (np.log(1-np.squeeze(np.asarray(outputActivations[inputxDex])))))).sum(axis=0))
# 	print("J:", J[inputxDex])
# print("\nJ total do dataset (com regularizacao):", (np.sum(J)/len(J))+regJSum)
# print("\n\n-------------------------------------------")

# grad = []
# for inputxDex in range(len(inputs)):
# 	gradIndex = 0
# 	index = len(activations[inputxDex])-1
# 	print("Rodando backpropagation")
# 	print("Calculando gradientes")
# 	activationsPL = activations[inputxDex][index-1]
# 	weightNoBias = weightLines[index-1]
# 	deltaNoBias = outputActivations[inputxDex] - outputs[inputxDex]
# 	print("delta", len(layers), ":", deltaNoBias.T)
# 	print("Gradientes de Theta", index)
# 	gradtmp = np.multiply(activationsPL, deltaNoBias.T).T
# 	if inputxDex == 0:
# 		grad.append(gradtmp)
# 	else:
# 		grad[gradIndex] = grad[gradIndex] + gradtmp
# 	print(gradtmp)

# 	while (index > 1):
# 		gradIndex = gradIndex+1
# 		delta = np.multiply(np.multiply(weightNoBias,deltaNoBias).sum(axis=0), np.multiply(activationsPL, (1-activationsPL)).T)
# 		deltaNoBias = delta[0,1:len(np.squeeze(np.asarray(delta)))]
# 		print("delta", index, ":", deltaNoBias)
# 		temp = delta[0,]
# 		index = index-1
# 		activationsPL = activations[inputxDex][index-1]
# 		weightNoBias = weightLines[index-1]
# 		print("Gradientes de Theta", index)
# 		gradtmp = np.multiply(activationsPL, deltaNoBias).T
# 		if inputxDex == 0:
# 			grad.append(gradtmp)
# 		else:
# 			grad[gradIndex] = grad[gradIndex] + gradtmp
# 		print(gradtmp)
# 		deltaNoBias = deltaNoBias.T

# weightsNoBias = []
# for wLayer in range(len(weightLines)):
# 	teste = np.asmatrix(weightLines[wLayer])
# 	teste[:,0] = 0
# 	weightsNoBias.append(regFactor*teste)

# grad.reverse()
# grad = np.add(grad, weightsNoBias)
# grad = 1/len(inputs)*grad
# print("Dataset completo processado. Calculando gradientes regularizados")
# for wLayer in range(len(grad)):
# 	print("Gradientes finais para Theta", wLayer+1, "(com regularizacao):")
# 	print(grad[wLayer])
# print("\n\n--------------------------------------------")
# eps = 0.000001
# print("Rodando verificacao numerica de gradientes (epsilon=",eps, ")")

# JPeps = copy.deepcopy(weightLines)
# JNeps = copy.deepcopy(weightLines)
# Jeps = copy.deepcopy(weightLines)
# for wLayer in range(len(weightLines)):
# 	for wRow in range(len(weightLines[wLayer])):
# 		for wElement in range(len(weightLines[wLayer][wRow])):
# 			weightTemp = copy.deepcopy(weightLines)
# 			weightTemp[wLayer][wRow][wElement] = weightTemp[wLayer][wRow][wElement]+eps
# 			JPeps[wLayer][wRow][wElement] = forwardProp(inputs, outputs, weightTemp, regFactor)
# 			weightTemp = copy.deepcopy(weightLines)
# 			weightTemp[wLayer][wRow][wElement] = weightTemp[wLayer][wRow][wElement]-eps
# 			JNeps[wLayer][wRow][wElement] = forwardProp(inputs, outputs, weightTemp, regFactor)
# 			Jeps[wLayer][wRow][wElement] = ((JPeps[wLayer][wRow][wElement] - JNeps[wLayer][wRow][wElement])/(2*eps))
# 	print("Gradiente numerico de Theta", wLayer+1, ":")
# 	print(np.asmatrix(Jeps[wLayer]))

# print("\n\n--------------------------------------------")
# print("Verificando corretude dos gradientes com base nos gradientes numericos:")
# for wLayer in range(len(Jeps)):
# 	print("Erro entre gradiente via backprop e gradiente numerico para Theta",wLayer+1,":")
# 	print(np.sum(np.absolute(grad[wLayer] - Jeps[wLayer])))