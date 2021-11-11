# NOTE: to run this code, use "python3 crf.py". 
# Running this code might require downloading a few libraries

# https://towardsdatascience.com/pos-tagging-using-crfs-ea430c5fb78b#1c6a
# https://towardsdatascience.com/implementing-a-linear-chain-conditional-random-field-crf-in-pytorch-16b0b9c4b4ea
# https://github.com/mtreviso/linear-chain-crf/blob/865bcf25fb33f73d59978426eb1c0f587e1f95f8/crf.py#L235
# https://www.overleaf.com/project/6176f620aee46f0817d1339f



import autograd.numpy as np
from autograd import grad
import math, time, string, copy
from functools import reduce
from conllu.models import TokenList, Token
from conllu import parse_incr
from prettytable import PrettyTable

def pretty_print(pos, lamda):
	labels = copy.deepcopy(pos)
	labels.insert(0, "")
	x = PrettyTable(labels)
	a = [["%.2f" % number for number in lamda[i]] for i in range(len(lamda))]
	[a[i].insert(0,pos[i]) for i in range(len(a))]
	#lamda = np.insert(lamda,0,pos, axis=1 )
	for row in a:
		x.add_row(row)
	print(x)

np.random.seed(0)
data_train_file = open("UD_English-EWT/en_ewt-ud-train.conllu", "r", encoding="utf-8")
data_dev_file = open("UD_English-EWT/en_ewt-ud-dev.conllu", "r", encoding="utf-8")
data_test_file = open("UD_English-EWT/en_ewt-ud-test.conllu", "r", encoding="utf-8")

train_X = []
train_Y = []
test_X = []
test_Y = []

num_types = 15 # means number of features including bos and eos

def word2features(word):
	result = np.zeros(num_types)
	result[2] = 1 if word[0].isdigit() else 0
	result[3] = 1 if word[0].isupper() else 0
	result[4] = 1 if '-' in word else 0
	result[5] = 1 if word in string.punctuation else 0
	result[6] = 1 if word[-3:].lower()=="ogy" else 0
	result[7] = 1 if word[-2:].lower()=="ed" else 0
	result[8] = 1 if word[-1:].lower()=="s" else 0
	result[9] = 1 if word[-2:].lower()=="ly" else 0
	result[10] = 1 if word[-3:].lower()=="ion" else 0
	result[11] = 1 if word[-4:].lower()=="tion" else 0
	result[12] = 1 if word[-3:].lower()=="ity" else 0
	result[13] = 1 if word[-3:].lower()=="ies" else 0
	result[14] = 1 if word[-3:].lower()=="ing" else 0
	return result

pos = set({})
pos.add("BOS")
pos.add("EOS")

bos = np.zeros(num_types)
bos[0]=1

eos = np.zeros(num_types)
eos[1]=1

if True:
	num_train = 5
	counter = 0
	for tokenlist in parse_incr(data_train_file):
		if counter == num_train:
			break
		X = []
		Y = []
		X.append(bos)
		Y.append("BOS")
		for token in tokenlist:
			word = token['form']
			part = token['upostag']
			features = word2features(word)
			X.append(features)
			Y.append(part)
			pos.add(part)
		X.append(eos)
		Y.append("EOS")
		train_X.append(X)
		train_Y.append(Y)
		counter += 1
else:
	print("you fool, you didn't implement me yet")
	exit()

pos = list(pos)
num_pos = len(pos)

pos_e = {pos[i]:i for i in range(num_pos)}
pos_d = {v: k for k, v in pos_e.items()}

# pos to pos
lamda = np.random.rand(num_pos, num_pos)

# type to pos
mu = np.random.rand(num_types, num_pos)

# transitions matrix
#M = np.random.rand(sent_len, num_pos, num_pos)
#M = np.zeros((sent_len, num_pos, num_pos))


# creates a one-hot vector, where each index corresponds to 
# "a certain pos y1 followed by a certain pos y2"
def f(l, y1, y2):
	return l[pos_e[y1], pos_e[y2]]
	
# creates a one-hot vector, where each index corresponds to 
# "a certain type x being of a certain pos y"
def g(m, x, y2):
	a = np.matmul(x, m)
	return a[pos_e[y2]]

def Mi(x, y1, y2, l, m):
	return np.exp(f(l,y1,y2) + g(m, x, y2))

def p(sentence, labels, theta):
	l = theta[0]
	m = theta[1]
	M = np.array([[[Mi(x, y1, y2, l, m) for y2 in pos] for y1 in pos] for x in sentence])
	var1 = math.prod([ M[i] [pos_e[labels[i-1]]] [pos_e[labels[i]]] for i in range(1, len(sentence))])
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
	backward[sen_len-1][pos_e["EOS"]] = 1
	for i in range(sen_len-2, 0, -1):
		backward[i] = np.dot(backward[i+1], M[i+1])

	prob = np.multiply(forward, backward)
	result = [pos_d[np.argmax(prob[i])] for i in range(1,sen_len-1)]
	return result

dodt = grad(obj_fun,2)

num_epoch = 100
for i in range(num_epoch):
	lamda[pos_e["EOS"]] = np.zeros(num_pos) 	# from EOS to anything is zero
	lamda[:, pos_e["BOS"]] = np.zeros(num_pos) 	# from anything to BOS is zero
	#lamda[pos_e["PUNCT"], pos_e["PUNCT"]] = 0

	# mu[:, pos_e["BOS"]] = np.zeros(num_types) 	# nothing belongs to BOS POS
	# mu[:, pos_e["EOS"]] = np.zeros(num_types)	# nothing belongs to EOS POS
	# mu[types_e[""]] = np.zeros(num_pos) 		# "" doesn't belong to any POS
	# mu[types_e[""], pos_e[""]] = 1 				# except ""
	# mu[types_e["BOS"]] = np.zeros(num_pos)      # BOS doesn't belong to any POS
	# mu[types_e["BOS"], pos_e["BOS"]] = 1 		# except BOS

	theta = [lamda, mu]
	#print(mu)
	#obj_fun(train_X, train_Y, theta)
	
	dT = dodt(train_X, train_Y, theta)
	lamda = np.add(theta[0], 0.1*dT[0])
	mu = np.add(theta[1], 0.1*dT[1])
	#continue
	actual = train_Y[0][1:-1]
	predicted = decode(train_X[0], lamda, mu)
	accuracy = sum([1 if actual[i] == predicted[i] else 0 for i in range(0,len(predicted))])/len(predicted)
	#pretty_print(pos,lamda)
	#print("actual:", actual)
	print("predicted:", predicted)
	print("accuracy:", accuracy)

	
	

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