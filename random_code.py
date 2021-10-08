from conllu.models import TokenList, Token
from conllu import parse_incr

data_train_file = open("UD_English-EWT/en_ewt-ud-train.conllu", "r", encoding="utf-8")
data_dev_file = open("UD_English-EWT/en_ewt-ud-dev.conllu", "r", encoding="utf-8")

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
