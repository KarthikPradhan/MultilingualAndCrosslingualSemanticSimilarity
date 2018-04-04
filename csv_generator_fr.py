from nltk.tokenize import PunktSentenceTokenizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import wordnet
from nltk import pos_tag
import numbers
import nltk
import csv, random

fd = open("data/french_moby.txt")
corpus = fd.read()

cust_sent_tokenize = PunktSentenceTokenizer(corpus)
corpus_arr = cust_sent_tokenize.tokenize(corpus)

# define the training capacity
train_set_capacity = round((80 * len(corpus_arr))/100)

train_data_arr = corpus_arr[:train_set_capacity]
test_data_arr = corpus_arr[train_set_capacity:]

# train_data = ",".join(x for x in train_data_arr)
# test_data = ",".join(y for y in test_data_arr)

def penn_to_wn(tag):
    """ Convert between a Penn Treebank tag to a simplified Wordnet tag """
    if tag.startswith('N'):
        return 'n'

    if tag.startswith('V'):
        return 'v'

    if tag.startswith('J'):
        return 'a'

    if tag.startswith('R'):
        return 'r'

    return None


def tagged_to_synset_fr(word):

    try:
        lemmas = wordnet.lemmas(word, lang="fra")
        hypernyms = lemmas[0].synset().hypernyms()
        return hypernyms
    except:
        return None

def sentence_similarity_fra(sentence, mth):
    """ compute the sentence similarity using Wordnet """
    sentence2 = []
    sentence1 = word_tokenize(sentence)
    # random.shuffle(corpus_arr)
    for w in corpus_arr[:12]:
        sentence2 = sentence2 + word_tokenize(w)

    # Get the synsets for the tagged words
    synsets1 = [tagged_to_synset_fr(tagged_word) for tagged_word in sentence1]
    synsets2 = [tagged_to_synset_fr(tagged_word) for tagged_word in sentence2]


    # Filter out the Nones
    synsets1 = [ss[0] for ss in synsets1 if ss]
    synsets2 = [ss[0] for ss in synsets2 if ss]

    score, count = 0.0, 0

    # For each word in the first sentence
    for synset in synsets1:
        # Get the similarity value of the most similar word in the other sentence
        if mth == 'path':
            scores = [synset.path_similarity(ss) for ss in synsets2]
        elif mth == 'wu':
            scores = [synset.wup_similarity(ss) for ss in synsets2]

        numscore=[x for x in scores if isinstance(x, numbers.Number)]
        if len(numscore)!=0:
            best_score=max(numscore)
            # Check that the similarity could have been computed
            if best_score is not None:
                score += best_score

        else:
            score += 0
        count += 1

    # Average the values
    if count == 0:
        return 0
    else:
        score /= count
        return score

# convert percentage into scores
def score_generator(score):
    if score >= 0 and score <= 0.2:
        return 0
    elif score > 0.2 and score <= 0.4:
        return 1
    elif score > 0.4 and score <= 0.6:
        return 2
    elif score > 0.6 and score <= 0.8:
        return 3
    else:
        return 4


def model_gen_train(sentences):
            with open('csv/fr/train/sim_fr.csv', 'w') as csvfile:
                fieldnames = ['Sentences', 'Score']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for sentence in sentences:
                    path_sem_sim = sentence_similarity_fra(sentence, 'wu')
                    writer.writerow({'Sentences': sentence, 'Score': score_generator(path_sem_sim)})

def model_gen_test(sentences):
            with open('csv/fr/test/sim_fr.csv', 'w') as csvfile:
                 fieldnames = ['Sentences', 'Score']
                 writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                 writer.writeheader()
                 for sentence in sentences:
                     path_sem_sim = sentence_similarity_fra(sentence, 'wu')
                     writer.writerow({'Sentences': sentence, 'Score': score_generator(path_sem_sim)})


model_gen_train(train_data_arr)
model_gen_test(test_data_arr)


print("CSV files created!")
