# NOTE: to run this code, use "python3 crf.py". 
# Running this code might require downloading a few libraries

# https://towardsdatascience.com/pos-tagging-using-crfs-ea430c5fb78b#1c6a
# https://towardsdatascience.com/implementing-a-linear-chain-conditional-random-field-crf-in-pytorch-16b0b9c4b4ea
# https://github.com/mtreviso/linear-chain-crf/blob/865bcf25fb33f73d59978426eb1c0f587e1f95f8/crf.py#L235
# https://www.overleaf.com/project/6176f620aee46f0817d1339f



import autograd.numpy as np
from autograd import grad
import math
from functools import reduce
from conllu.models import TokenList, Token
from conllu import parse_incr
import time
import copy
from prettytable import PrettyTable

np.random.seed(0)
data_train_file = open("UD_English-EWT/en_ewt-ud-train.conllu", "r", encoding="utf-8")
data_dev_file = open("UD_English-EWT/en_ewt-ud-dev.conllu", "r", encoding="utf-8")
data_test_file = open("UD_English-EWT/en_ewt-ud-test.conllu", "r", encoding="utf-8")

train_X = []
train_Y = []
test_X = []
test_Y = []

types = set({})
types.add("BOS")
types.add("")

pos = set({})
pos.add("BOS")
pos.add("")

num_train = 10000
num_test = 0

if True:
	counter = 0
	for tokenlist in parse_incr(data_train_file):
		if counter == num_train:
			break
		X = []
		Y = []
		X.append("BOS")
		Y.append("BOS")
		for token in tokenlist:
			word = token['form']
			part = token['upostag']
			X.append(word)
			Y.append(part)
			types.add(word)
			pos.add(part)
		X.append("")
		Y.append("")
		train_X.append(X)
		train_Y.append(Y)
		counter += 1

	counter = 0
	for tokenlist in parse_incr(data_test_file):
		if counter == num_test:
			break
		X = []
		Y = []
		X.append("BOS")
		Y.append("BOS")
		for token in tokenlist:
			word = token['form']
			part = token['upostag']
			X.append(word)
			Y.append(part)
			types.add(word)
			pos.add(part)
		X.append("")
		Y.append("")
		test_X.append(X)
		test_Y.append(Y)
		counter += 1
else:
	train_X.append(["BOS","We","won","!", ""])
	train_Y.append(["BOS", "pronoun", "verb", "punc", ""])
	types = set(train_X[0])
	pos = set(train_Y[0])
	
types = list(types)
pos = list(pos)
num_types = len(types)
num_pos = len(pos)

types_e = {types[i]:i for i in range(num_types)}
types_d = {v: k for k, v in types_e.items()}
pos_e = {pos[i]:i for i in range(num_pos)}
pos_d = {v: k for k, v in pos_e.items()}

# pos to pos
#lamda = np.random.rand(num_pos, num_pos)

# type to pos
#mu = np.random.rand(num_types, num_pos)

# transitions matrix
#M = np.random.rand(sent_len, num_pos, num_pos)
#M = np.zeros((sent_len, num_pos, num_pos))

def print_lamda(pos, l):
	labels = copy.deepcopy(pos)
	labels.insert(0, "None")
	x = PrettyTable(labels)
	a = [["%.2E" % number for number in l[i]] for i in range(len(l))]
	[a[i].insert(0,pos[i]) for i in range(len(a))]
	for row in a:
		x.add_row(row)
	print(x)

def print_mu(types, pos, m):
	labels = copy.deepcopy(pos)
	labels.insert(0, "None")
	x = PrettyTable(labels)
	a = [["%.2E" % number for number in m[i]] for i in range(len(m))]
	[a[i].insert(0,types[i]) for i in range(len(a))]
	for row in a:
		x.add_row(row)
	print(x)

def print_forward(sentence, pos, forward):
	labels = copy.deepcopy(pos)
	labels.insert(0, "None")
	x = PrettyTable(labels)
	a = [["%.2E" % number for number in forward[i]] for i in range(len(forward))]
	[a[i].insert(0,sentence[i]) for i in range(len(a))]
	for row in a:
		x.add_row(row)
	print(x)	

# creates a one-hot vector, where each index corresponds to 
# "a certain pos y1 followed by a certain pos y2"
def f(l, y1, y2):
	return l[pos_e[y1], pos_e[y2]]
	
# creates a one-hot vector, where each index corresponds to 
# "a certain type x being of a certain pos y"
def g(m, x, y2):
	return m[types_e[x], pos_e[y2]]

def Mi(x, y1, y2, l, m):
	return np.exp(f(l,y1,y2)*g(m, x, y2))

def p(sentence, labels, theta):
	l = theta[0]
	m = theta[1]
	M = np.array([[[Mi(x, y1, y2, l, m) for y1 in pos] for y2 in pos] for x in sentence])
	print(M)
	var1 = math.prod([ M[i] [pos_e[labels[i-1]]] [pos_e[labels[i]]] for i in range(1, len(sentence))])
	var2 = reduce(np.matmul, M)
	print(var2)
	return var1/var2[0, num_pos-1]
	
def obj_fun(sentences, labels, theta):
	return sum([np.log(p(sentences[i], labels[i], theta)) for i in range(len(sentences))])
	
def decode2(sentence, lamda, mu):
	sen_len = len(sentence)
	M = np.array([[[Mi(x, y1, y2, lamda, mu) for y2 in pos] for y1 in pos] for x in sentence])
	print_lamda(pos,M[1])
	print(sentence)

	forward = np.zeros((sen_len, num_pos))
	forward[0][pos_e["BOS"]] = 1
	for i in range(1, sen_len-1):
		forward[i] = np.dot(forward[i-1], M[i])

	print_forward(sentence, pos, forward)

	backward = np.zeros((sen_len, num_pos))
	backward[sen_len-1][pos_e[""]] = 1
	for i in range(sen_len-2, 0, -1):
		backward[i] = np.dot(M[i+1], backward[i+1])

	print_forward(sentence, pos, backward)

	prob = np.multiply(forward, backward)
	print_forward(sentence, pos, prob)
	result = [pos_d[np.argmax(prob[i])] for i in range(1,sen_len-1)]
	return result



lamda = np.zeros((num_pos,num_pos))
mu = np.zeros((num_types,num_pos))

for j in range(len(train_X)):
	x = train_X[j]
	y = train_Y[j]
	for i in range(len(x)):
		if i == 0:
			continue
		lamda[pos_e[y[i-1]]][pos_e[y[i]]] += 1
		mu[types_e[x[i]]][pos_e[y[i]]] += 1

print_lamda(pos,lamda)
print_mu(types,pos,mu)

actual = train_Y[0][1:-1]
predicted = decode2(train_X[0], lamda, mu)
accuracy = sum([1 if actual[i] == predicted[i] else 0 for i in range(0,len(predicted))])/len(predicted)
print("actual:", actual)
print("predicted:", predicted)
print("accuracy:", accuracy)


# num_epoch = 100
# for i in range(num_epoch):
# 	lamda[pos_e[""]] = np.zeros(num_pos) 		# from "" to anything is zero
# 	lamda[:, pos_e["BOS"]] = np.zeros(num_pos) 	# from anything to BOS is zero

# 	mu[:, pos_e["BOS"]] = np.zeros(num_types) 	# nothing belongs to BOS POS
# 	mu[:, pos_e[""]] = np.zeros(num_types)		# nothing belongs to "" POS
# 	mu[types_e[""]] = np.zeros(num_pos) 		# "" doesn't belong to any POS
# 	mu[types_e[""], pos_e[""]] = 1 				# except ""
# 	mu[types_e["BOS"]] = np.zeros(num_pos)      # BOS doesn't belong to any POS
# 	mu[types_e["BOS"], pos_e["BOS"]] = 1 		# except BOS

# 	theta = [lamda, mu]
# 	#print(mu)
# 	#obj_fun(train_X, train_Y, theta)
# 	dodt = grad(obj_fun,2)
# 	dT = dodt(train_X, train_Y, theta)
# 	lamda = np.add(theta[0], 0.1*dT[0])
# 	mu = np.add(theta[1], n0.1*dT[1])
# 	#continue
# 	actual = train_Y[0][1:-1]
# 	predicted = decode(train_X[0], lamda, mu)
# 	accuracy = sum([1 if actual[i] == predicted[i] else 0 for i in range(0,len(predicted))])/len(predicted)
# 	#print("actual:", actual)
# 	print("predicted:", predicted)
# 	print("accuracy:", accuracy)

	
	

#dpdt = grad(p,2)
#r = dpdt(train_X[0], train_Y[0], [lamda, mu])

# problems: super slow , can I use sklearn.preprocessing.OneHotEncoder, trying it for a small sentence first and then expandind it to the whole training set means some variable names were not changed, which caused problems

# next steps: 
# setup crf from towarddatascience and test performance: https://github.com/mtreviso/linear-chain-crf
# just to see if its worth taking a deep dive in and following in their footsteps
# set up overleaf for this and copy over what has to be in MS1
# document progress so far - what you have now is good enough for MS1
# might be worth a look but try an prebuilt gradient optimzer
# python3 -m cProfile crf.py > profile1.txt