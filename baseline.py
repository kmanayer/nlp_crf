from statistics import mode
from functools import reduce
from conllu.models import TokenList, Token
from conllu import parse_incr
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def print_scores(label, true, pred):
	acc = accuracy_score(true, pred)
	macro_p = precision_score(true, pred, average="weighted", zero_division=0)
	macro_r = recall_score(true, pred, average="weighted", zero_division=0)
	macro_f1 = f1_score(true, pred, average="weighted", zero_division=0)
	print("------------------------------------------------")
	print("The model evaluated on the " + label.upper() + " dataset")
	print("                Accuracy: %1.3f" % acc)
	print("Weighted Macro Precision: %1.3f" % (macro_p))
	print("   Weighted Macro Recall: %1.3f" % (macro_r))
	print("       Weighted Macro F1: %1.3f" % (macro_f1))
	print("------------------------------------------------")

def flatten(l):
	return sum(l, [])

data_test_file = open("UD_English-EWT/en_ewt-ud-test.conllu", "r", encoding="utf-8")


true = []
predicted = []
num_test = 5

counter = 0


for tokenlist in parse_incr(data_test_file):
	if counter == num_test:
		break
	true_ = []
	for token in tokenlist:
		true_.append(token['upostag'])
	p = mode(true_)
	predicted_ = [p] * len(tokenlist)
	true.append(true_)
	predicted.append(predicted_)
	counter += 1


true = flatten(true)
predicted = flatten(predicted)

print_scores("baseline test", true, predicted)



