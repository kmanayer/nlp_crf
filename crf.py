import math, time, sys, pickle
import autograd.numpy as np
from autograd import grad
from functools import reduce
from conllu.models import TokenList, Token
from conllu import parse_incr
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# likelihood of "a certain pos y1 followed by a certain pos y2"
def f(l, y1, y2):
	return l[pos_e[y1], pos_e[y2]]
	
# likelihood of "a certain type x being of a certain pos y"
def g(m, x, y2):
	return m[types_e[x], pos_e[y2]]

# returns one entry of one frame in the overall M matrix
def Mi(x, y1, y2, l, m):
	return np.exp(f(l,y1,y2) + g(m, x, y2))

def p(sentence, labels, theta):
	l = theta[0] # lambda
	m = theta[1] # mu
	M = np.array([[[Mi(x, y1, y2, l, m) for y2 in pos] for y1 in pos] for x in sentence])
	var1 = math.prod([ M[i] [pos_e[labels[i-1]]] [pos_e[labels[i]]] for i in range(1, len(sentence))])
	var2 = reduce(np.matmul, M)
	return var1/var2[0, num_pos-1]
	
def obj_fun(sentences, labels, theta):
	return sum([np.log(p(sentences[i], labels[i], theta)) for i in range(len(sentences))])
	
def decode(sentence, lamda, mu):
	sen_len = len(sentence)
	M = np.array([[[Mi(x, y1, y2, lamda, mu) for y2 in pos] for y1 in pos] for x in sentence])
	
	forward = np.zeros((sen_len, num_pos))
	forward[0][pos_e["BOS"]] = 1
	for i in range(1, sen_len-1):
		forward[i] = np.dot(forward[i-1], M[i])

	backward = np.zeros((sen_len, num_pos))
	backward[sen_len-1][pos_e[""]] = 1
	for i in range(sen_len-2, 0, -1):
		backward[i] = np.dot(M[i+1], backward[i+1])

	prob = np.multiply(forward, backward)
	result = [pos_d[np.argmax(prob[i])] for i in range(1,sen_len-1)]
	return result
	
def flatten(l):
	return sum(l, [])

def inspect_lamda(l,c):
	result = ""
	lf = l.flatten()
	(a,b) = l.shape
	# neat trick from https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
	ind = np.argpartition(lf, -c)[-c:]
	ind = ind[np.argsort(lf[ind])]
	for i in reversed(ind):
		p1 = i%b
		p2 = int(i/b)
		p1 = pos_d[p1]
		p2 = pos_d[p2]
		print("POS: {0:8} POS: {1:8} value: {2:.3f}".format(p2, p1,lf[i]))
		result += p2 + ", " + p1 + ", " + "{0:.3f},\n".format(lf[i])
	print(result)

def inspect_mu(m,c):
	result = ""
	mf = m.flatten()
	(a,b) = m.shape
	# neat trick from https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
	ind = np.argpartition(mf, -c)[-c:]
	ind = ind[np.argsort(mf[ind])]
	for i in reversed(ind):
		p = i%b
		w = int(i/b)
		p = pos_d[p]
		w = types_d[w]
		print("Word: {0:15} POS: {1:8} value: {2:.3f}".format(w, p, mf[i]))
		result += w + ", " + p + ", " + "{0:.3f},\n".format(mf[i])
	print(result)

def print_scores(label, true, pred):
	acc = accuracy_score(true, pred)
	macro_p = precision_score(true, pred, average="weighted", zero_division=0)
	macro_r = recall_score(true, pred, average="weighted", zero_division=0)
	macro_f1 = f1_score(true, pred, average="weighted", zero_division=0)
	graph_file.write("{0:.3f}, {1:.3f}, {2:.3f}, {3:.3f}, ".format(acc, macro_p, macro_r, macro_f1))
	print("------------------------------------------------")
	print("The model evaluated on the " + label.upper() + " dataset")
	print("                Accuracy: %1.3f" % acc)
	print("Weighted Macro Precision: %1.3f" % (macro_p))
	print("   Weighted Macro Recall: %1.3f" % (macro_r))
	print("       Weighted Macro F1: %1.3f" % (macro_f1))
	print("------------------------------------------------")

np.random.seed(0)
data_train_file = open("UD_English-EWT/en_ewt-ud-train.conllu", "r", encoding="utf-8")
data_dev_file = open("UD_English-EWT/en_ewt-ud-dev.conllu", "r", encoding="utf-8")
data_test_file = open("UD_English-EWT/en_ewt-ud-test.conllu", "r", encoding="utf-8")

train_X = []
train_Y = []
dev_X = []
dev_Y = []
test_X = []
test_Y = []

types = set({})
types.add("BOS")
types.add("")

pos = set({})
pos.add("BOS")
pos.add("")

num_train = 50
num_dev = 5
num_test = 5

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
for tokenlist in parse_incr(data_dev_file):
	if counter == num_dev:
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
	dev_X.append(X)
	dev_Y.append(Y)
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
	
types = list(types)
pos = list(pos)
num_types = len(types)
num_pos = len(pos)

# encoders and decoders
types_e = {types[i]:i for i in range(num_types)}
types_d = {v: k for k, v in types_e.items()}
pos_e = {pos[i]:i for i in range(num_pos)}
pos_d = {v: k for k, v in pos_e.items()}

# pos to pos, misspelled on purpose since lambda is a keyword
lamda = np.random.rand(num_pos, num_pos)

# type to pos
mu = np.random.rand(num_types, num_pos)

testing = False
if len(sys.argv) == 3 and sys.argv[1] == "test":
	testing = True
	
inspect = False
if len(sys.argv) == 4 and sys.argv[1] == "inspect":
	inspect = True

dodt = grad(obj_fun,2)

file_name = "crf_" + str(num_train) + ".model"
graph_file_name = "crf_" + str(num_train) + ".graph"
graph_file = open(graph_file_name, "w")

#lr = 1/num_train
lr = 0.01
num_epoch = 1000
for i in range(num_epoch):
	if testing or inspect:
		break;
	start = time.time()
	lamda[pos_e[""]] = np.zeros(num_pos) 		# from "" to anything is zero
	lamda[:, pos_e["BOS"]] = np.zeros(num_pos) 	# from anything to BOS is zero

	mu[:, pos_e["BOS"]] = np.zeros(num_types) 	# nothing belongs to BOS POS
	mu[:, pos_e[""]] = np.zeros(num_types)		# nothing belongs to "" POS
	mu[types_e[""]] = np.zeros(num_pos) 		# "" doesn't belong to any POS
	mu[types_e[""], pos_e[""]] = 1 				# except ""
	mu[types_e["BOS"]] = np.zeros(num_pos)      # BOS doesn't belong to any POS
	mu[types_e["BOS"], pos_e["BOS"]] = 1 		# except BOS

	theta = [lamda, mu]
	llh = obj_fun(train_X, train_Y, theta)
	dT = dodt(train_X, train_Y, theta)
	lamda = np.add(theta[0], lr*dT[0])
	mu = np.add(theta[1], lr*dT[1])
	
	print("epoch ", i+1, ", log-likelihood:", llh)
	graph_file.write("{0:3}, {1:.3f}, ".format(i, llh))
	
	actual = flatten([train_Y[i][1:-1] for i in range(num_train)])
	predicted = flatten([decode(train_X[i], lamda, mu) for i in range(num_train)])
	print_scores("training", actual, predicted)
	
	actual = flatten([dev_Y[i][1:-1] for i in range(num_dev)])
	predicted = flatten([decode(dev_X[i], lamda, mu) for i in range(num_dev)])
	print_scores("dev", actual, predicted)

	graph_file.write("\n")
	graph_file.flush()

	output_file = open(file_name, "wb")
	pickle.dump([lamda, mu, types, pos, types_e, pos_e, types_d, pos_d], output_file)

	print(time.time() - start, " secs for ", num_train, " training, and ", num_dev, " dev")
	
if testing:
	file_name = sys.argv[2]
	output_file = open(file_name, "rb")
	model = pickle.load(output_file)
	lamda = model[0]
	mu = model[1]
	types = model[2]
	pos = model[3]
	types_e = model[4]
	pos_e = model[5]
	types_d = model[6]
	pos_d = model[7]
	actual = flatten([test_Y[i][1:-1] for i in range(num_test)])
	predicted = flatten([decode(test_X[i], lamda, mu) for i in range(num_test)])
	print_scores("test", actual, predicted)

if inspect:
	file_name = sys.argv[2]
	num_to_inspect = int(sys.argv[3])
	output_file = open(file_name, "rb")
	model = pickle.load(output_file)
	lamda = model[0]
	mu = model[1]
	types = model[2]
	pos = model[3]
	types_e = model[4]
	pos_e = model[5]
	types_d = model[6]
	pos_d = model[7]
	inspect_lamda(lamda,num_to_inspect)
	inspect_mu(mu,num_to_inspect)

