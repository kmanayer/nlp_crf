import numpy as np
from autograd import grad
import math
from functools import reduce
from conllu.models import TokenList, Token
from conllu import parse_incr

data_train_file = open("UD_English-EWT/en_ewt-ud-train.conllu", "r", encoding="utf-8")

train_X = []
train_Y = []

for tokenlist in parse_incr(data_train_file):
	X = []
	Y = []
	X.append("BOS")
	Y.append("BOS")
	for token in tokenlist:
		X.append(token['form'])
		Y.append(token['xpostag'])
	X.append("")
	Y.append("")
	train_X.append(X)
	train_Y.append(Y)
	break

print(train_X)
print(train_Y)
exit()
	
	
types = ['BOS', 'Arya', 'is', 'a', 'nice', 'kitty', '!', '']
sentence = types
pos = ['BOS', 'Proper Noun', 'Verb', 'Article', 'Adjective', 'Noun', 'Punctuation', '']
labels = pos

num_types = len(types)
sent_len = len(types)
num_pos = len(pos)

types_encoded = {types[i]:i for i in range(num_types)}
pos_encoded = {pos[i]:i for i in range(num_pos)}

# pos to pos
#lamda = np.zeros((num_pos*num_pos))
lamda_in = np.random.rand(num_pos*num_pos) * 0.1
#print(lamda)

# type to pos
#mu = np.zeros((num_types*num_pos))
mu_in = np.random.rand(num_types*num_pos) * 0.1
#print(mu)

# transitions matrix
#M = np.random.rand(sent_len, num_pos, num_pos)
M = np.zeros((sent_len, num_pos, num_pos))


# creates a one-hot vector, where each index corresponds to "a certain pos y1 followed by a certain pos y2"
def f(y1, y2):
	var1 = np.zeros((num_pos, num_pos));
	var1[pos_encoded[y1], pos_encoded[y2]] = 1
	return var1.flatten()
	
# creates a one-hot vector, where each index corresponds to "a certain type x being of a certain pos y"
def g(x, y2):
	var1 = np.zeros((num_types, num_pos))
	var1[types_encoded[x], pos_encoded[y2]] = 1
	return var1.flatten()


def Mi(x, y1, y2, lamda, mu):
	return np.exp(np.dot(lamda, f(y1, y2) + np.dot(mu, g(x, y2))))
	
def p(sentence, labels, theta):
	lamda = theta[0]
	mu = theta[1]
	M = np.array([[[Mi(x, y1, y2, lamda, mu) for y1 in pos] for y2 in pos] for x in types])
	var1 = math.prod([ M[i][i-1][i] for i in range(1, sent_len)])
	var2 = reduce(np.matmul, M)
	return var1/var2[0, num_pos-1]
	
print(p(sentence, labels, [lamda_in, mu_in]))
dpdt = grad(p,2)
print(dpdt(sentence, labels, [lamda_in, mu_in]))


def obj_fun(X,Y,T):
	print("hello")
	
	

