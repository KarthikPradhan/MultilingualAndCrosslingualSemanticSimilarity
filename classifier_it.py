import csv
from collections import defaultdict
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk, random
from nltk.tokenize import PunktSentenceTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LogisticRegression

# Load the files
fd = open("data/italian_moby.txt")
corpus = fd.read()

# tokenizing
cust_sent_tokenize = PunktSentenceTokenizer(corpus)
corpus_arr = cust_sent_tokenize.tokenize(corpus)
word_corp_token = word_tokenize(corpus)

stop_words = set(stopwords.words("italian"))

d = defaultdict(list)

# load the sentence and similarity score from the file
# open the file in universal line ending mode
test_corp = ""
with open('csv/it/train/sim_it.csv', 'rU') as infile:
    # read the file as a dictionary for each row ({header : value})
    reader = csv.DictReader(infile)
    data = {}
    for row in reader:
        for header, value in row.items():
            if header == "Sentences": test_corp = test_corp + value
            try:
                data[header].append(value)
            except KeyError:
                data[header] = [value]

# Get the sentences and scores from training set
for key in data:
    if key == 'Sentences':
        sent = data[key]
    elif key == 'Score':
        score = data[key]

# get feature sets to train
i = 0
t = []
tup = ()
list_of_tuples = []
while (i < len(score)):
    sent = [w for w in sent if w not in stop_words]
    if score[i] != '':
        # d[str(score[i])].append(word_tokenize(str(sent[i])))
        wd_arr = word_tokenize(str(sent[i]))
        for val in wd_arr:
            fs = {}
            fs["word"] = val
            # fs = generate_fs(word_tokenize(str(sent[i])))
            list_of_tuples = list_of_tuples + [(fs, score[i])]
    i += 1

# get testset
with open('csv/it/test/sim_it.csv', 'rU') as infile_t:
    # read the file as a dictionary for each row ({header : value})
    reader_t = csv.DictReader(infile_t)
    data_t = {}
    for row in reader_t:
        for header, value in row.items():
            try:
                data_t[header].append(value)
            except KeyError:
                data_t[header] = [value]

# Get the sentences and scores from test set
for key_t in data:
    if key_t == 'Sentences':
        sent_t = data_t[key_t]
    elif key_t == 'Score':
        score_t = data_t[key_t]

test_set = []
j = 0
while (j < len(score_t)):
    sent_t = [w for w in sent_t if w not in stop_words]
    if score_t[j] != '':
        # d[str(score[i])].append(word_tokenize(str(sent[i])))
        wd_arr = word_tokenize(str(sent_t[j]))
        for val in wd_arr:
            fs = {}
            fs["word"] = val
            # fs = generate_fs(word_tokenize(str(sent[i])))
            test_set = test_set + [(fs, score_t[j])]
    j += 1

# print(list_of_tuples)
print("\nClassifier results:")



# Naive Bayes
naive_bayes = nltk.NaiveBayesClassifier.train(list_of_tuples)
print("Naive Bayes Algorithm Accuracy of Feature Set Path doc: ", (nltk.classify.accuracy(naive_bayes, test_set)) * 100)
print(naive_bayes.show_most_informative_features(15))

# Logistic Regression
LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(list_of_tuples)
print("LogisticRegression_classifier Algo accuracy:",
      (nltk.classify.accuracy(LogisticRegression_classifier, test_set)) * 100)
