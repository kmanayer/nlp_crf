import math, time, sys, pickle, string
import autograd.numpy as np
from autograd import grad
from functools import reduce
from conllu.models import TokenList, Token
from conllu import parse_incr
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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
	
def flatten(l):
	return sum(l, [])

def f(l, y1, y2):
	return l[pos_e[y1], pos_e[y2]]
	
def g(m, x, y2):
	return np.dot(x,m[:,pos_e[y2]])
	#a = np.matmul(x, m)
	#return a[pos_e[y2]]

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
	M = np.array([[[Mi(x, y1, y2, lamda, mu) for y2 in pos] for y1 in pos] for x in sentence])
	
	forward = np.zeros((sen_len, num_pos))
	forward[0][pos_e["BOS"]] = 1
	for i in range(1, sen_len-1):
		forward[i] = np.dot(forward[i-1], M[i])

	backward = np.zeros((sen_len, num_pos))
	backward[sen_len-1][pos_e["EOS"]] = 1
	for i in range(sen_len-2, 0, -1):
		backward[i] = np.dot(M[i+1], backward[i+1])

	prob = np.multiply(forward, backward)
	result = [pos_d[np.argmax(prob[i])] for i in range(1,sen_len-1)]
	return result

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

num_types = 15 # means number of features including bos and eos

types_d ={	0: 'BOS', 
			1: 'EOS', 
			2: 'number', 
			3: 'upper', 
			4: 'hyphen', 
			5: 'punct', 
			6: '-ogy', 
			7: '-ed', 
			8: '-s', 
			9: '-ly', 
			10: '-ion', 
			11: '-tion', 
			12: '-ity', 
			13: '-ies', 
			14: '-ing'}

pos = set({})
pos.add("BOS")
pos.add("EOS")

bos = np.zeros(num_types)
bos[0]=1

eos = np.zeros(num_types)
eos[1]=1

num_train = 51
num_dev = 5
num_test = 5

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

counter = 0
for tokenlist in parse_incr(data_dev_file):
	if counter == num_dev:
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
	dev_X.append(X)
	dev_Y.append(Y)
	counter += 1

counter = 0
for tokenlist in parse_incr(data_test_file):
	if counter == num_test:
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
	test_X.append(X)
	test_Y.append(Y)
	counter += 1

pos = list(pos)
num_pos = len(pos)

# encoder and decoder
pos_e = {pos[i]:i for i in range(num_pos)}
pos_d = {v: k for k, v in pos_e.items()}

# pos to pos
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

file_name = "crf2_" + str(num_train) + ".model"
graph_file_name = "crf2_" + str(num_train) + ".graph"
graph_file = open(graph_file_name, "w")

#lr = 1/num_train
lr = 0.1
num_epoch = 100
for i in range(num_epoch):
	if testing or inspect:
		break;
	start = time.time()
	lamda[pos_e["EOS"]] = np.zeros(num_pos) 	# from EOS to anything is zero
	lamda[:, pos_e["BOS"]] = np.zeros(num_pos) 	# from anything to BOS is zero

	mu[:, pos_e["BOS"]] = np.zeros(num_types) 	# nothing belongs to BOS POS
	mu[:, pos_e["EOS"]] = np.zeros(num_types)	# nothing belongs to EOS POS

	theta = [lamda, mu]
	llh = obj_fun(train_X, train_Y, theta)
	dT = dodt(train_X, train_Y, theta)
	lamda = np.add(theta[0], 0.05*dT[0])
	mu = np.add(theta[1], 0.05*dT[1])
	
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
	pickle.dump([lamda, mu, pos, pos_e, pos_d], output_file)

	print(time.time() - start, " secs for ", num_train, " training, and ", num_test, " dev")
	#pretty_print(pos,lamda)

if testing:
	file_name = sys.argv[2]
	output_file = open(file_name, "rb")
	model = pickle.load(output_file)
	lamda = model[0]
	mu = model[1]
	pos = model[2]
	pos_e = model[3]
	pos_d = model[4]
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
	pos = model[2]
	pos_e = model[3]
	pos_d = model[4]
	inspect_lamda(lamda,num_to_inspect)
	inspect_mu(mu,num_to_inspect)
	
