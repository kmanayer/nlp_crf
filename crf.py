import autograd.numpy as np
from autograd import grad
import math
from functools import reduce
from conllu.models import TokenList, Token
from conllu import parse_incr
import time

np.random.seed(0)
data_train_file = open("UD_English-EWT/en_ewt-ud-train.conllu", "r", encoding="utf-8")
#data_dev_file = open("UD_English-EWT/en_ewt-ud-dev.conllu", "r", encoding="utf-8")
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

if False:
	num_train = 1
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
			part = token['xpostag']
			X.append(word)
			Y.append(part)
			types.add(word)
			pos.add(part)
		X.append("")
		Y.append("")
		train_X.append(X)
		train_Y.append(Y)
		counter += 1

	num_test = 0
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
			part = token['xpostag']
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

#print(types)
#print(pos)

# pos to pos
lamda = np.random.rand(num_pos, num_pos)
#print(lamda)

# type to pos
mu = np.random.rand(num_types, num_pos)
#print(mu)

#exit()
# transitions matrix
#M = np.random.rand(sent_len, num_pos, num_pos)
#M = np.zeros((sent_len, num_pos, num_pos))


# creates a one-hot vector, where each index corresponds to "a certain pos y1 followed by a certain pos y2"
def f(l, y1, y2):
	return l[pos_e[y1], pos_e[y2]]
	#var1 = np.zeros((num_pos, num_pos));
	#var1[pos_e[y1], pos_e[y2]] = 1
	#return var1.flatten()
	
# creates a one-hot vector, where each index corresponds to "a certain type x being of a certain pos y"
def g(m, x, y2):
	return m[types_e[x], pos_e[y2]]
	#var1 = np.zeros((num_types, num_pos))
	#var1[types_e[x], pos_e[y2]] = 1
	#return var1.flatten()


def Mi(x, y1, y2, l, m):
	results = np.exp(f(l,y1,y2) + g(m, x, y2))
	#print(results)
	return results
	#return np.exp(np.dot(l, f(y1, y2)) + np.dot(m, g(x, y2)))
	#return np.exp(0)
	
def p(sentence, labels, theta):
	l = theta[0]
	m = theta[1]
	M = np.array([[[Mi(x, y1, y2, l, m) for y1 in pos] for y2 in pos] for x in sentence])
	var1 = math.prod([ M[i][pos_e[labels[i-1]]][pos_e[labels[i]]] for i in range(1, len(sentence))])
	var2 = reduce(np.matmul, M)
	return var1/var2[0, num_pos-1]
	
def obj_fun(sentences, labels, theta):
	return sum([np.log(p(sentences[i], labels[i], theta)) for i in range(len(sentences))])
	
def decode(sentence, lamda, mu):
	sen_len = len(sentence)
	M = np.array([[[Mi(x, y1, y2, lamda, mu) for y1 in pos] for y2 in pos] for x in sentence])
	
	forward = np.zeros((sen_len, num_pos))
	forward[0][pos_e["BOS"]] = 1
	for i in range(1, sen_len-1):
		forward[i] = np.dot(forward[i-1], M[i])

	backward = np.zeros((sen_len, num_pos))
	backward[sen_len-1][pos_e[""]] = 1
	for i in range(sen_len-2, 0, -1):
		backward[i] = np.dot(backward[i+1], M[i+1])

	prob = np.multiply(forward, backward)
	result = [pos_d[np.argmax(prob[i])] for i in range(1,sen_len-1)]
	return result


num_epoch = 1
for i in range(num_epoch):
	# lamda[pos_e[""]] = np.zeros(num_pos)
	# lamda[:, pos_e["BOS"]] = np.zeros(num_pos)

	# mu[types_e[""]] = np.zeros(num_pos)
	# mu[types_e["BOS"]] = np.zeros(num_pos)
	# mu[:, pos_e["BOS"]] = np.zeros(num_types)
	# mu[:, pos_e[""]] = np.zeros(num_types)
	# mu[types_e["BOS"], pos_e["BOS"]] = 1
	# mu[types_e[""], pos_e[""]] = 1

	theta = [lamda, mu]
	#obj_fun(train_X, train_Y, theta)
	dodt = grad(obj_fun,2)
	dT = dodt(train_X, train_Y, theta)
	lamda = np.add(theta[0], 0.1*dT[0])
	mu = np.add(theta[1], 0.1*dT[1])
	#continue
	actual = train_Y[0][1:-1]
	predicted = decode(train_X[0], lamda, mu)
	accuracy = sum([1 if actual[i] == predicted[i] else 0 for i in range(0,len(predicted))])/len(predicted)
	#print("actual:", actual)
	print("predicted:", predicted)
	print("accuracy:", accuracy)

	
	

#dpdt = grad(p,2)
#r = dpdt(train_X[0], train_Y[0], [lamda, mu])

	
# problems: super slow , can I use sklearn.preprocessing.OneHotEncoder, trying it for a small sentence first and then expandind it to the whole training set means some variable names were not changed, which caused problems

