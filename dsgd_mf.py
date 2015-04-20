from pyspark import SparkContext
import numpy as np
import functools
import random
import operator
import sys


# used for labeling the id on row in W in parallel
def labelkeyRow(x):
	global sizePerRowBlock
	return x[0]/sizePerRowBlock

# used for labeling the id on column in H in parallel
def labelkeyCol(arg,x):
	global sizePerColBlock
	global blockSize
	pid = x[0]/sizePerColBlock + arg
	if pid>= blockSize:
		return pid - blockSize
	else:
		return pid

# running SGD algorithm
def dsgd(nowIter,iterator):
	i=0
	partW = {}
	partH = {}
	global V
	global dictRowCount
	global dictColCount
	global lamb
	tao0 = 100
	global beta
	stepSize = pow(100+nowIter , -1*beta)
	updateIter = 0

	for keyPair in iterator:				
		partitionID = keyPair[0]
		val = keyPair[1]
		if i == 0:
			Wk = list(val[0])
			for pair in Wk:
				partW[pair[0]] = pair[1]
		tempH = val[1]
		x = tempH[0]
		hVec = tempH[1]
		for y in partW.keys():
			keyYX = `y`+","+`x`
			if V.has_key(keyYX):
				updateIter += 1
				val = V[keyYX] - np.dot(partW[y],hVec)
				tempA = []
				for i in range(0,len(hVec)):
				#	please use following line, if you don't want to do L2 regulization
				#	tempA.append(hVec[i]*eps*(-2)*val)
					tempA.append(stepSize*(hVec[i]*eps*(-2)*val+2*lamb*(partW[y][i]/dictRowCount[y])))
				Wp = map(summ,partW[y],tempA)
				tempA = []		
				for i in range(0,len(partW[y])):
				#	please use following line, if you don't want to do L2 regulization
				#	tempA.append(partW[y][i]*eps*(-2)*val)#
					tempA.append(stepSize*(partW[y][i]*eps*(-2)*val+2*lamb*(hVec[i]/dictColCount[x])))
				hVec = map(summ,hVec,tempA)
				partW[y] = Wp
		partH[x] = hVec
		i=i+1
	yield [[partW],[partH],updateIter]	
		
def summ(ele,ele2):
	return ele-ele2

# parsing data with csv format
def preprocessMapSubSampling (x):
	val = x.split('\n')
	for row in val:
		rowSplit = row.split(',')
		if len(rowSplit)>1:
			yield [rowSplit[0], rowSplit[1], rowSplit[2]]

# parsing data with netflix data format
def preprocessMap (x):
	keyValuePair = x.split(':')
	ValueSplit = keyValuePair[1].split('\n')
	listI = []
	for row in ValueSplit:
		rowSplit = row.split(',')
		if len(rowSplit)>1:
			yield [keyValuePair[0],rowSplit[0],rowSplit[1]]	

factors = int(sys.argv[1])
blockSize = int(sys.argv[2])
numIter = int(sys.argv[3])
beta = float(sys.argv[4])
lamb = float(sys.argv[5])
inputV = sys.argv[6]
fwFileName = sys.argv[7]
fhFileName = sys.argv[8]

import time

start = time.time()

dictRow = {}
dictCol = {}
V = {}
sc = SparkContext("local", "Simple App")
txtFile = sc.wholeTextFiles(inputV)

# preprocssing data into rdd
xt = txtFile.values().flatMap(preprocessMapSubSampling).collect()

dictRowCount = {}
dictColCount = {}

maxCol = 0
maxRow = 0
for tuple in xt:	
	y = int(tuple[0])
	x = int(tuple[1])
	z = tuple[2]
	if dictRowCount.has_key(y):
        dictRowCount[y]=dictRowCount[y] + 1
	else:
		dictRowCount[y] = 1
	
	if dictColCount.has_key(x):
        	dictColCount[x] = dictColCount[x] + 1
	else:
		dictColCount[x] = 1	
	if y>maxRow:
		maxRow = y
	if x>maxCol:
		maxCol = x
	V[`y`+","+`x`] = float(z)

rowSize = maxRow
colSize = maxCol

Wlen= len(dictRow)-1
Hlen = len(dictCol)-1
eps = 0.01
W = {}
H = {}
inilist = []

#initialize W and H
for y in range(0,maxRow+1):
	inilist = []
	for i in range(factors):
		inilist.append(random.random())
	W[y] = list(inilist)

for x in range(0,maxCol+1):
	inilist = []
	for i in range(factors):
		inilist.append(random.random())
	H[x] = list(inilist)

sizePerRowBlock = rowSize/blockSize
if (rowSize%blockSize!=0):
	sizePerRowBlock+=1

sizePerColBlock = colSize/blockSize
if (colSize%blockSize!=0):
	sizePerColBlock+=1

totalIter = 0

reconErrorList =[]

# start to run DSGD-MF
for numOfIteration in range(0,numIter):
	for k2 in range(0,blockSize): 
		rddRow = sc.parallelize(W.items(),blockSize).keyBy(labelkeyRow)
		rddCol = sc.parallelize(H.items(),blockSize).keyBy(functools.partial(labelkeyCol,k2))
		r= rddRow.groupByKey()
		t = r.join(rddCol)
		v=t.mapPartitions(functools.partial(dsgd,totalIter)).collect()
		
		for block in v: 
			updateWPairs = block[0]
			W.update(updateWPairs[0])
			updateHPairs = block[1]
			H.update(updateHPairs[0])
			totalIter += int(block[2])
	loss = 0
	for key in V.keys():
		yx=key.split(",")
		y = yx[0]
		x = yx[1]	
		val = V[key] - np.dot(W[int(y)],H[int(x)]) 
		loss = loss + val * val
	reconErrorList.append(loss/len(V))

# sorted data for outputing the model of W, H 
sorted_w = sorted(W.items(),key = operator.itemgetter(0))
sorted_h = sorted(H.items(),key = operator.itemgetter(0))

fw = open (fwFileName,'w')
t=0
for pair in sorted_w:
	if t!=0:
		fw.write(','.join(map(str,pair[1])))
		fw.write('\n')
	t+=1

fw.close()

fh = open (fhFileName,'w')
t=0
strlist = []
for pair in sorted_h:
	if t!=0:
		it=0
		for num in pair[1]:
			if t==1:
				strlist.append("")
			strlist[it]+=str(num)+","
			it+=1			
	t+=1

for it in range(0,factors):
	fh.write(strlist[it][:-1]+'\n')

fh.close()
end = time.time()

elapsed = end - start

print "timeUsed"
print elapsed