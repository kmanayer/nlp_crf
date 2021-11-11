from conllu.models import TokenList, Token
from conllu import parse_incr

data_train_file = open("UD_English-EWT/en_ewt-ud-train.conllu", "r", encoding="utf-8")
data_dev_file = open("UD_English-EWT/en_ewt-ud-dev.conllu", "r", encoding="utf-8")
data_test_file = open("UD_English-EWT/en_ewt-ud-test.conllu", "r", encoding="utf-8")

if False:
	# trying to figure out which tokens to use for start and stop
	counter = 0
	for tokenlist in parse_incr(data_train_file):
		for token in tokenlist:
			word = token["form"]
			counter += 1
			if "BOS" == word or "" == word:
				print("nooo", word)
	print("training", counter)
				
	counter = 0
	for tokenlist in parse_incr(data_dev_file):
		for token in tokenlist:
			word = token["form"]
			counter += 1
			if "" == word or "" == word:
				print("double nooo", word)
	print("testing", counter)
	
def helpme(values):
	return ''.join([str(value) for value in values])

if False:
	X = ["random/en_train.txt", "random/en_dev.txt", "random/en_test.txt"]
	Y = [data_train_file, data_dev_file, data_test_file]
	Z = [100, 10, 10]
	for i in range(3):
		f = open(X[i], "w")
		f.write("ID\tWORD\tTAG\n")
		counter = 0
		for tokenlist in parse_incr(Y[i]):
			if counter > Z[i]:
				continue
			for token in tokenlist:
				idx = token["id"]
				idx = str(idx) if type(idx) == int else helpme(idx)
				word = token["form"]
				word = "'" if "\"" == word or "\"" == word else word
				tag = token["xpostag"]
				tag = "'" if "''" == tag or "``" == tag else tag
				f.write(idx + "\t" + word + "\t" + tag + "\n")
			f.write("\t\t\n")
			counter += 1
		f.close()

if True:
	POS1 = set({})
	POS2 = set({})
	for tokenlist in parse_incr(data_train_file):
		for token in tokenlist:
			pos1 = token["upostag"]
			POS1.add(pos1)
			pos2 = token["xpostag"]
			POS2.add(pos2)

	print("training number of different univeral POS:", len(POS1))
	print(POS1)
	print("training number of different lang specific POS:", len(POS2))
	print(POS2)
	