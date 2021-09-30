import autograd.numpy as np
from autograd import grad
import math
from functools import reduce


types = ['BOS', 'Arya', 'is', 'a', 'nice', 'kitty', '!', 'EOS']
sentence = types
pos = ['BOS', 'Proper Noun', 'Verb', 'Article', 'Adjective', 'Noun', 'Punctuation',  'EOS']
labels = pos

num_types = len(types)
sent_len = len(types)
num_pos = len(pos)

types_encoded = {types[i]:i for i in range(num_types)}
pos_encoded = {pos[i]:i for i in range(num_pos)}

# pos to pos
#lamda = np.zeros((num_pos*num_pos))
lamda = np.random.rand(num_pos*num_pos) * 0.1
#print(lamda)

# type to pos
#mu = np.zeros((num_types*num_pos))
mu = np.random.rand(num_types*num_pos) * 0.1
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
	
print(p(sentence, labels))

def obj_fun(X,Y,T):
	print("hello")
	
	

