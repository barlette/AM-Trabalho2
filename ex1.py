import sys
import numpy as np
import pandas as pd
import csv
import math
import time
import random
import copy

from numpy import linalg as LA
def divideIntoKFolds(kFolds, fileName, classIndex):
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
        Classifications.append(int(flist[classIndex-1]))
        dataset.append(flist)

    #print(set(Classifications))
    num_of_classes = len(set(Classifications))
    #print(num_of_classes)

    norm = np.zeros(shape=(np.size(dataset,0),np.size(dataset,1)))
    #norm[:,0] = np.asarray(dataset)[:,0]

    for column in range(np.size(dataset,1)):
    	norm[:,column] = normalize(np.asmatrix(dataset)[0:, column]).T

    #print("norm", norm)
    #Vamos percorrer a lista de dados e separar em n pilhas de acordo com a classe
    dct = {}
    instances_list = []
    classes = np.unique(Classifications)
    #print(classes)
    for nClass in range(0,num_of_classes):
        for nIndex in range(0, len(norm)):
            if Classifications[nIndex] == classes[nClass]:
                instances_list.append(norm[nIndex])
        dct['%s' % nClass] = instances_list

        instances_list = []

    # print('Creating kFolds...')
    #print("dct", dct)
    # print(random.randint(1,4))
    dataFolds = []
    tempList = []
    randomIndex = 0
    for nClass in range(0,num_of_classes):
        instances_list = dct.get('%s' %nClass)
        #print(instances_list)
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
    #print("dataFolds", dataFolds)
    return dataFolds

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def normalize(df):
	if(df.max() - df.min() == 0):
		return np.zeros(df.shape)
	else:
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
	#z = []
	outputActivations = []
	print("Testando:\n")
	for inputxDex in range(len(inputs)):
		# print("Conjunto de treinamento:")
		# print("x:", np.asmatrix(inputs[inputxDex]).T)
		# print("y:", outputs[inputxDex])
		# print("--------------------------------------------")
		# print("Calculando erro/custo J da rede")
		# print("Propagando entrada", np.asmatrix(inputs[inputxDex]).T)
		activations.append([])
		activations[inputxDex].append(inputs[inputxDex].T)
		#z.append([])
		#z[inputxDex].append(inputs[inputxDex].T)
		for layer in range(len(weightLines)):
			# print(weightLines[layer])
			# print(activations[inputxDex][layer])
			# print(np.matmul(weightLines[layer], activations[inputxDex][layer]))
			activations[inputxDex].append(sigmoid(np.matmul(weightLines[layer], activations[inputxDex][layer])))
			#z[inputxDex].append(np.matmul(weightLines[layer], activations[inputxDex][layer]))
			activations[inputxDex][layer+1] = np.vstack([1,activations[inputxDex][layer+1]])
			#z[inputxDex][layer+1] = np.vstack([1,z[inputxDex][layer+1]])
			# print("z",layer,":",z[inputxDex][layer].T)
			# print("a",layer,":",activations[inputxDex][layer].T)
		index = len(activations[inputxDex])-1
		outputActivations.append(activations[inputxDex][index][1:len(activations[inputxDex][index])])
		#outputZ = z[inputxDex][index][1:len(z[inputxDex][index])]
		# print("z",len(z[inputxDex]),":",outputZ.T)
		# print("a",len(activations[inputxDex]),":",outputActivations[inputxDex].T)
		print(np.asarray(inputs[inputxDex]), ";", np.squeeze(np.asarray(outputActivations[inputxDex])))
		#print("Saida predita:", np.squeeze(np.asarray(outputActivations[inputxDex])))
		#print("Saida esperada:", np.squeeze(np.asarray(outputs[inputxDex])))
		for classPredict in range(len(outputActivations[inputxDex])):
			if classPredict == 0:
				J.append((-np.multiply(np.squeeze(np.asarray(outputs[inputxDex])), (np.log(np.squeeze(np.asarray(outputActivations[inputxDex][classPredict]))))) -np.multiply((1-np.squeeze(np.asarray(outputs[inputxDex]))),  (np.log(1-np.squeeze(np.asarray(outputActivations[inputxDex][classPredict])))))).sum(axis=0))
			else:
				J[inputxDex] = J[inputxDex]+(-np.multiply(np.squeeze(np.asarray(outputs[inputxDex])), (np.log(np.squeeze(np.asarray(outputActivations[inputxDex][classPredict]))))) -np.multiply((1-np.squeeze(np.asarray(outputs[inputxDex]))),  (np.log(1-np.squeeze(np.asarray(outputActivations[inputxDex][classPredict])))))).sum(axis=0)
		#print("J:", J[inputxDex])
	outputsTypes = np.unique(np.asarray(outputs))
	#print(outputsTypes)
	classOutput = np.zeros((len(outputs),1))
	#print(classOutput)
	dist = np.zeros(len(outputsTypes))

	for indexout in range(len(outputs)):
		# print("numero de exemplos de teste:", len(outputs))
		
		#print("Outputs Esperados:\n", outputs[indexout])
		#print("Outputs Previstos:\n", outputsTypes[np.argmax(outputActivations[indexout])])
		classOutput[indexout] = outputsTypes[np.argmax(outputActivations[indexout])]
	#print("Classificação final:\n", classOutput)
	# print("J final:", (np.sum(J)/len(J))+regJSum)
	ftable = np.zeros((len(outputsTypes), len(outputsTypes)))
	#print(ftable)
	#print(outputs)
	fMeasure = np.zeros(len(outputsTypes))
	recall = np.zeros(len(outputsTypes))
	precision = np.zeros(len(outputsTypes))

	for indexout in range(len(outputs)):
		row = np.where(outputsTypes == classOutput[indexout])
		#print(row)
		column = np.where(outputsTypes == outputs[indexout])
		#print(column[1])
		ftable[row,column[1]] = ftable[row,column[1]]+1
	
	for indexout in range(len(outputsTypes)):
		#print(np.sum(ftable, axis=0))
		#print(np.sum(ftable, axis=1))
		recall[indexout] = ftable[indexout, indexout]/(np.sum(ftable, axis=0)[indexout])
		precision[indexout] = ftable[indexout, indexout]/(np.sum(ftable, axis=1)[indexout])
		#print(ftable[indexout, indexout])
		#print(recall[indexout])
		#print(precision[indexout])
		fMeasure[indexout] = (2*precision[indexout]*recall[indexout])/(precision[indexout]+recall[indexout])
	fMeasure[np.isnan(fMeasure)] = 0
	#print(fMeasure)
	#print(ftable)
	#print("Final FMeasure:", np.sum(fMeasure)/len(fMeasure))
	return (np.sum(fMeasure)/len(fMeasure))

def forwardPropNum(inputs, outputs, weightLines, regFactor):
	regJ = []
	regJSum = 0
	for(layer) in range(len(weightLines)):
		regJ.append(np.power(weightLines[layer],2))
		regJSum = regJSum + np.sum(regJ[layer][0:len(regJ[layer]),1:])
		
	regJSum = (regFactor/(2*len(inputs)))*regJSum
	J = []
	activations = []
	#z = []
	outputActivations = []
	#print(len(inputs))
	for inputxDex in range(len(inputs)):
		#print("Conjunto de treinamento:")
		#print("x:", np.asmatrix(inputs[inputxDex]).T)
		#print("y:", outputs[inputxDex])
		#print("--------------------------------------------")
		#print("Calculando erro/custo J da rede")
		#print("Propagando entrada", np.asmatrix(inputs[inputxDex]).T)
		activations.append([])
		activations[inputxDex].append(np.asmatrix(inputs[inputxDex]).T)
		#z.append([])
		#z[inputxDex].append(np.asmatrix(inputs[inputxDex]).T)
		for layer in range(len(weightLines)):
			#print(weightLines[layer])
			#print(activations[inputxDex][layer])
			#print(sigmoid(np.asmatrix(weightLines[layer]).dot(activations[inputxDex][layer])))
			activations[inputxDex].append(sigmoid(np.dot(weightLines[layer], activations[inputxDex][layer])))
			#z[inputxDex].append(np.matmul(weightLines[layer], activations[inputxDex][layer]))
			activations[inputxDex][layer+1] = np.vstack([1,activations[inputxDex][layer+1]])
			#z[inputxDex][layer+1] = np.vstack([1,z[inputxDex][layer+1]])
			#print("z",layer,":",z[inputxDex][layer].T)
			#print("a",layer,":",activations[inputxDex][layer].T)
		index = len(activations[inputxDex])-1
		outputActivations.append(activations[inputxDex][index][1:len(activations[inputxDex][index])])
		#outputZ = z[inputxDex][index][1:len(z[inputxDex][index])]
		#print("z",len(z[inputxDex]),":",outputZ.T)
		#print("a",len(activations[inputxDex]),":",outputActivations[inputxDex].T)
		#print("Saida predita:", np.squeeze(np.asarray(outputActivations[inputxDex])))
		#print("Saida esperada:", np.squeeze(np.asarray(outputs[inputxDex])))
		J.append((-np.multiply(np.squeeze(np.asarray(outputs[inputxDex])), (np.log(np.squeeze(np.asarray(outputActivations[inputxDex]))))) -np.multiply((1-np.squeeze(np.asarray(outputs[inputxDex]))),  (np.log(1-np.squeeze(np.asarray(outputActivations[inputxDex])))))).sum(axis=0))
	#print("retorno da função:", ((np.sum(J)/len(J))+regJSum))
	return((np.sum(J)/len(J))+regJSum)
		



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

#for layer in range(len(initialWeightLines)):
	#print("Theta",layer+1,"inicial (pesos de cada neuronio, incluindo bias, armazenados nas linhas):")
	#for row in range(len(initialWeightLines[layer])):
		#print(initialWeightLines[layer][row])

numberFolds = 10
#grad = []




for inc in range(0, 1):
	weightLines = []
	for f in range(0, numberFolds):
		weightLines.append(initialWeightLines)
	print("\nFator de regularizacao:", regFactor)
	for ln in range(0, 1):
		folds = divideIntoKFolds(numberFolds, sys.argv[3], int(sys.argv[4]))
		fMeasures = np.zeros(len(folds))
		#print(folds[1])
		fMeasures = np.zeros(len(folds))
		for nFold in range(len(folds)):
			print("\nFold:", nFold)
			#print(folds[nFold])
			#fold de teste
			testing = copy.deepcopy(folds[nFold])
			#o resto para treinamento
			inputsTotal = np.delete(folds, nFold)
			inputsTotal = np.concatenate(inputsTotal[0:])
			#print(inputsTotal[0:50,:])
			#as saídas esperadas
			#print("inputsTotal:", inputsTotal)
			outputs = copy.deepcopy(np.asmatrix(inputsTotal[0:, int(sys.argv[4])-1]).T)
			#print(outputs)
			outputsTypes = np.unique(np.asarray(outputs))
			#print(outputsTypes)
			outputMatrix = np.zeros((len(outputs), len(outputsTypes)))
			for outMatrix in range(len(outputs)):
				column = np.where(outputsTypes == outputs[outMatrix])
				#print(outputs[outMatrix])
				#print(column)
				outputMatrix[outMatrix, column[1]] = 1

			outputMatrix = outputMatrix

			#print(outputMatrix)

			#print("outputs:", outputs)
			#inserindo o bias
			if (int(sys.argv[4])-1) != 0:
				inputsTotal = np.delete(inputsTotal, int(sys.argv[4])-1, axis=1)
				inputsTotal = np.insert(inputsTotal, 0, 1, axis=1)
			else:
				inputsTotal[0:, 0] = np.ones(len(inputsTotal))

			print("\nConjunto de treinamento (9 folds) com o bias inserido:\n", inputsTotal)

			minibatch = 1
			batchIndex = 0
			batchSize = 32
			print("\nTamanho do minibatch:", batchSize)
			while minibatch==1:
				grad = []			
				if(((batchIndex+1)*batchSize) < len(inputsTotal)):
					inputs = inputsTotal[batchIndex*batchSize:((batchIndex+1)*batchSize),:]
					outputs = outputMatrix[batchIndex*batchSize:((batchIndex+1)*batchSize),:]
				else:
					inputs = inputsTotal[batchIndex*batchSize:len(inputsTotal),:]
					outputs = outputMatrix[batchIndex*batchSize:len(inputsTotal),:]
					minibatch = 0

				print("\nMinibatch: ", batchIndex)
				#print(inputs)
				#print(outputs)
				batchIndex = batchIndex+1
				regJ = []
				regJSum = 0
				for(layer) in range(len(weightLines[nFold])):
					for row in range(len(weightLines[nFold][layer])):
						#print(row)
						if row==0:
							regJ.append(np.power(np.array(weightLines[nFold][layer][row]),2))
						else:
							regJ[layer] = np.vstack((regJ[layer], np.power(np.array(weightLines[nFold][layer][row]),2)))
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
					for layer in range(len(weightLines[nFold])):
						#print(weightLines[nFold][layer])
						#print(activations[inputxDex][layer])
						#print(sigmoid(np.asmatrix(weightLines[nFold][layer]).dot(activations[inputxDex][layer])))
						activations[inputxDex].append(sigmoid(np.dot(weightLines[nFold][layer], activations[inputxDex][layer])))
						z[inputxDex].append(np.dot(weightLines[nFold][layer], activations[inputxDex][layer]))
						activations[inputxDex][layer+1] = np.vstack([1,activations[inputxDex][layer+1]])
						z[inputxDex][layer+1] = np.vstack([1,z[inputxDex][layer+1]])
						#print("z",layer,":",z[inputxDex][layer].T)
						#print("a",layer,":",activations[inputxDex][layer].T)
					index = len(activations[inputxDex])-1
					outputActivations.append(activations[inputxDex][index][1:len(activations[inputxDex][index])])
					outputZ = z[inputxDex][index][1:len(z[inputxDex][index])]
					#print("z",len(z[inputxDex])-1,":",outputZ.T)
					#print("a",len(activations[inputxDex])-1,":",outputActivations[inputxDex].T)
					#print("Saida predita:", np.squeeze(np.asarray(outputActivations[inputxDex])))
					#print("Saida esperada:", np.squeeze(np.asarray(outputs[inputxDex])))
					for classPredict in range(len(outputActivations[inputxDex])):
						if classPredict == 0:
							J.append((-np.multiply(np.squeeze(np.asarray(outputs[inputxDex])), (np.log(np.squeeze(np.asarray(outputActivations[inputxDex]))))) -np.multiply((1-np.squeeze(np.asarray(outputs[inputxDex]))),  (np.log(1-np.squeeze(np.asarray(outputActivations[inputxDex])))))).sum(axis=0))
						else:
							J[inputxDex] = J[inputxDex]+(-np.multiply(np.squeeze(np.asarray(outputs[inputxDex])), (np.log(np.squeeze(np.asarray(outputActivations[inputxDex]))))) -np.multiply((1-np.squeeze(np.asarray(outputs[inputxDex]))),  (np.log(1-np.squeeze(np.asarray(outputActivations[inputxDex])))))).sum(axis=0)
					#print("J:", J[inputxDex])
				#print("J total do dataset (com regularizacao):", (np.sum(J)/len(J))+regJSum)
				#print("\n\n-------------------------------------------")

				#grad = []
				for inputxDex in range(len(inputs)):
					#print(inputxDex)
					gradIndex = 0
					#print(len(activations[inputxDex]))
					index = len(activations[inputxDex])-1
					#print("Rodando backpropagation")
					#print("Calculando gradientes")
					
					activationsPL = activations[inputxDex][index-1]


					deltaNoBias = outputActivations[inputxDex].T - outputs[inputxDex]
					#print(outputActivations[inputxDex].T)
					#print(outputs[inputxDex])
					#print("Gradientes de Theta", index, ", input:", inputxDex)
					#print("activationsPL: ", activationsPL)
					#print("delta", len(layers), ":", deltaNoBias)
					gradtmp = np.multiply(deltaNoBias, activationsPL).T
					if inputxDex == 0:
						grad.append(gradtmp)
					else:
						grad[gradIndex] = grad[gradIndex] + gradtmp
					#print(gradtmp)

					deltaNoBias = deltaNoBias.T
					weightNoBias = weightLines[nFold][index-1]

					while (index > 1):
						#print("teste")
						gradIndex = gradIndex+1
						#print("weight no bias:", weightNoBias)
						#print("delta no bias:", deltaNoBias)
						#print("no sum: ", np.multiply(weightNoBias,deltaNoBias))
						#print(np.multiply(weightNoBias,deltaNoBias).sum(axis=0))
						#print("derivativeSig:", derivativeSig(activationsPL).T)
						delta = np.multiply(np.multiply(np.asmatrix(weightNoBias),deltaNoBias).sum(axis=0), derivativeSig(activationsPL).T)
						#print("delta: ", delta)
						deltaNoBias = delta[0,1:len(np.squeeze(np.asarray(delta)))]
						#print("delta no bias ", index, ":", deltaNoBias)
						temp = delta[0,]
						index = index-1
						activationsPL = activations[inputxDex][index-1]
						weightNoBias = weightLines[nFold][index-1]

						#print("activationsPL: ", activationsPL)
						gradtmp = np.multiply(activationsPL, deltaNoBias).T
						if inputxDex == 0:
							grad.append(gradtmp)
						else:
							grad[gradIndex] = grad[gradIndex] + gradtmp
						#print("Gradientes de Theta", index, ", input:", inputxDex)
						#print(gradtmp)
						deltaNoBias = deltaNoBias.T

				weightsNoBias = []
				for wLayer in range(len(weightLines[nFold])):
					teste = np.asmatrix(weightLines[nFold][wLayer])
					teste[:,0] = 0
					weightsNoBias.append(regFactor*teste)
				grad.reverse()
				grad = np.add(grad, weightsNoBias)
				#print("gradiente final:", grad)
				grad = grad/len(inputs)

				
				print("\nMinibatch processado. Calculando gradientes regularizados")
				for wLayer in range(len(grad)):
				 	#print(np.asmatrix(weightLines[nFold][wLayer]))
				 	print("\nGradientes finais para Theta", wLayer+1, "(com regularizacao):")
				 	print(grad[wLayer])
				print("\n\n--------------------------------------------")


				eps = 0.000001
				print("\nRodando verificacao numerica de gradientes (epsilon=",eps, ")")

				JPeps = []
				JNeps = []
				Jeps = []
				JPeps = copy.deepcopy(weightLines[nFold])
				JNeps = copy.deepcopy(weightLines[nFold])
				Jeps = copy.deepcopy(weightLines[nFold])
				weightTemp = []

				#print(len(weightLines[nFold]))
				#print(weightLines[nFold])
				for wLayer in range(len(weightLines[nFold])):
					for wRow in range(len(weightLines[nFold][wLayer])):
						#print(len(weightLines[nFold][wLayer]))
						for wElement in range(len(weightLines[nFold][wLayer][wRow])):
							#print(len(weightLines[nFold][wLayer][wRow]))
							weightTemp = copy.deepcopy(weightLines[nFold])
							weightTemp[wLayer][wRow][wElement] = weightTemp[wLayer][wRow][wElement]+eps
							#print("temp:", weightTemp)
							JPeps[wLayer][wRow][wElement] = forwardPropNum(inputs, outputs, weightTemp, regFactor)
							weightTemp = copy.deepcopy(weightLines[nFold])
							weightTemp[wLayer][wRow][wElement] = weightTemp[wLayer][wRow][wElement]-eps
							JNeps[wLayer][wRow][wElement] = forwardPropNum(inputs, outputs, weightTemp, regFactor)
							Jeps[wLayer][wRow][wElement] = ((JPeps[wLayer][wRow][wElement] - JNeps[wLayer][wRow][wElement])/(2*eps))
					print("Gradiente numerico de Theta", wLayer+1, ":")
					print(np.asmatrix(Jeps[wLayer]))

				#print("temp:", weightTemp)
				#print("JPeps:", JPeps)
				#print("JNeps:", JNeps)
				#print("Jeps:", Jeps)
				print("\n\n--------------------------------------------")
				print("\nVerificando corretude dos gradientes com base nos gradientes numericos:")
				for wLayer in range(len(Jeps)):
					print("\nErro entre gradiente via backprop e gradiente numerico para Theta",wLayer+1,":")
					print(np.sum(np.absolute(grad[wLayer] - Jeps[wLayer])))
					#print("norm:", LA.norm(grad[wLayer] - Jeps[wLayer])/LA.norm(grad[wLayer] + Jeps[wLayer]))

				#print("weightLines1", weightLines[nFold])
				weightLines[nFold] = weightLines[nFold]-0.5*grad
				for layer in range(len(weightLines[nFold])):
					weightLines[nFold][layer] = weightLines[nFold][layer].tolist()
				#print("weightLines2", weightLines[nFold])
				#print("Conjunto de teste:\n", testing)
				#print("testing:", testing[0:][0])
			inputTesting = np.asmatrix(testing)
			outputsTesting = copy.deepcopy(np.asmatrix(testing)[0:, int(sys.argv[4])-1])
				#print("outputsTesting:", outputsTesting)
			if (int(sys.argv[4])-1) != 0:
				inputTesting = np.delete(inputTesting, int(sys.argv[4])-1, axis=1)
				inputTesting = np.insert(inputTesting, 0, 1, axis=1)
			else:
				inputTesting[0:, 0] = np.ones((len(inputTesting),1))
			#print("input testing:", inputTesting)
			#print(weightLines[nFold])
			#print(forwardProp(inputTesting, outputsTesting, weightLines[nFold], regFactor))
			fMeasures[nFold] = forwardProp(inputTesting, outputsTesting, weightLines[nFold], regFactor)
		print("FMeasure para cada fold: \n", fMeasures)
		print("Média das FMeasure: \n", np.sum(fMeasures)/len(fMeasures))
		#print(np.sum(fMeasures)/len(fMeasures))
		#fMeasureTotal.append(fMeasures)
		#print("FMeasure Fold:", fMeasureTotal[ln]/len(fMeasureTotal[ln]))
			#print("Final Result:", np.sum(fMeasures)/len(fMeasures))
	#regFactor = regFactor+0.1
