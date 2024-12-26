from gensim.models import Word2Vec
from nltk import word_tokenize
import os
# define training data
# sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
#  ['this', 'is', 'the', 'second', 'sentence'],
#  ['yet', 'another', 'sentence'],
#  ['one', 'more', 'sentence'],
#  ['and', 'the', 'final', 'sentence']]

EMBEDDING_DIM = 100

current_dir = os.getcwd()

sentences = []
for line in open(current_dir + '/kamus/malang.txt', encoding='utf-8'):
    if '==' not in line:
        continue

    line_split = line.split('==')
    judul = line_split[0].lower()
    jawaban = line_split[1].lower()

    sentences.append(word_tokenize(judul))
    sentences.append(word_tokenize(jawaban))

# train model
model = Word2Vec(sentences, min_count=1, epochs=10, vector_size=EMBEDDING_DIM)
# summarize the loaded model
# print(model)
# summarize vocabulary
words = list(model.wv.index_to_key)
# print(words)
# access vector for one word
# print(model.wv.get_vector('sentence'))
# save model
# model.save('model.bin')
# load model
# new_model = Word2Vec.load('model.bin')
# print(new_model)

new_embed = []
for w in words:
    new_embed.append(w + ' ' + ' '.join([str(v) for v in model.wv.get_vector(w)]))

with open(current_dir + '/embedding/malang.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(new_embed))