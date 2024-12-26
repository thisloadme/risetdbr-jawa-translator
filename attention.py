from __future__ import print_function, division
from builtins import range, input

import os, sys
from keras.models import Model
from keras.layers import Input, LSTM, GRU, Dense, Embedding, Bidirectional, RepeatVector, Concatenate, Activation, Dot, Lambda
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import keras.backend as K

import numpy as np
import matplotlib.pyplot as plt
import os
import json

current_dir = os.getcwd()


# expected shape N x T x D
# N = jumlah sample, T = panjang sequence, D = dimensi vector
def softmax_over_time(x):
    assert(K.ndim(x) > 2)
    e = K.exp(x - K.max(x, axis=1, keepdims=True))
    s = K.sum(e, axis=1, keepdims=True)
    return e/s

# konfig
BATCH_SIZE = 64
EPOCHS = 300
LATENT_DIM = 256 # latent dimensionality of the encoding space
LATENT_DIM_DECODER = 256 # bedakan dengan encoder biar mantap
NUM_SAMPLES = 10000
MAX_SEQUENCE_LENGTH = 100
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100

input_texts = [] # sentence in ori language
target_texts = [] # sentence in target language
target_texts_inputs = [] # sentences in target language offset by 1

t = 0
for line in open(current_dir + '/kamus/malang.txt', encoding='utf-8'):
    t += 1
    if t > NUM_SAMPLES:
        break

    if '==' not in line:
        continue

    line_split = line.split('==')
    input_text = line_split[1]
    translation = line_split[0]

    target_text = translation + ' <eos>'
    target_text_input = '<sos> ' + translation

    input_texts.append(input_text)
    target_texts.append(target_text)
    target_texts_inputs.append(target_text_input)

# convert input ke int
tokenizer_inputs = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer_inputs.fit_on_texts(input_texts)
input_sequences = tokenizer_inputs.texts_to_sequences(input_texts)

# word index mapping
word2idx_inputs = tokenizer_inputs.word_index

# max input sequences
max_len_input = max(len(s) for s in input_sequences)

# convert output ke int
tokenizer_outputs = Tokenizer(num_words=MAX_NUM_WORDS, filters='')
tokenizer_outputs.fit_on_texts(target_texts + target_texts_inputs)
target_sequences = tokenizer_outputs.texts_to_sequences(target_texts)
target_sequences_inputs = tokenizer_outputs.texts_to_sequences(target_texts_inputs)

# word index mapping
word2idx_outputs = tokenizer_outputs.word_index

# add 1 idx
num_words_outputs = len(word2idx_outputs) + 1

# max input sequences
max_len_target = max(len(s) for s in target_sequences)

# pad the seq
encoder_inputs = pad_sequences(input_sequences, maxlen=max_len_input)
# print(encoder_inputs.shape)
# print(encoder_inputs[0])

decoder_inputs = pad_sequences(target_sequences_inputs, maxlen=max_len_target, padding='post')
# print(decoder_inputs.shape)
# print(decoder_inputs[0])

decoder_targets = pad_sequences(target_sequences, maxlen=max_len_target, padding='post')
# print(decoder_targets.shape)
# print(decoder_targets[0])

# load pre trained word vectors
word2vec = {}
with open(current_dir + '/embedding/malang.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vec = np.asarray(values[1:], dtype='float32')
        word2vec[word] = vec

# prepare embedding matrix
# +1 karena untuk sos/eos
num_words = min(MAX_NUM_WORDS, len(word2idx_inputs) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word2idx_inputs.items():
    if i < MAX_NUM_WORDS:
        embedding_vector = word2vec.get(word)
        if embedding_vector is not None:
            # kata yang ga ada di embedding index di set 0 semua
            embedding_matrix[i] = embedding_vector

# load pre trained word embeddings ke embedding layer
embedding_layer = Embedding(
    num_words,
    EMBEDDING_DIM,
    weights=[embedding_matrix],
    input_length=max_len_input
)

decoder_targets_one_hot = np.zeros(
    (
        len(input_texts),
        max_len_target,
        num_words_outputs
    ),
    dtype='float32'
)

# assign the values
for i, d in enumerate(decoder_targets):
    for t, word in enumerate(d):
        decoder_targets_one_hot[i, t, word] = 1

# build the model
encoder_inputs_placeholder = Input(shape=(max_len_input,))
x = embedding_layer(encoder_inputs_placeholder)
encoder = Bidirectional(LSTM(LATENT_DIM, return_sequences=True, dropout=0.5))
encoder_outputs = encoder(x)

# setup decoder using [h,c] as initial state
decoder_inputs_placeholder = Input(shape=(max_len_target,))

decoder_embedding = Embedding(num_words_outputs, EMBEDDING_DIM)
decoder_inputs_x = decoder_embedding(decoder_inputs_placeholder)


# ATTENTION HERE
# Attention layer harus global, karena akan di ulang terus menerus di decoder
attn_repeat_layer = RepeatVector(max_len_input)
attn_concat_layer = Concatenate(axis=-1)
attn_dense1 = Dense(10, activation='tanh')
attn_dense2 = Dense(1, activation=softmax_over_time)
attn_dot = Dot(axes=1) # untuk mengalikaan alpha dengan h

def one_step_attention(h, st_1):
    # h = h(1), .... , h(Tx), shape = (Tx, LATENT_DIM * 2)
    # st_1 = s(t-1), shape = (LATENT_DIM_DECODER,)

    # copy s(t-1) Tx times
    #now shape = (Tx, LATENT_DIM_DECODER)
    st_1 = attn_repeat_layer(st_1)

    # concatenate all h(t)'s with s(t-1)
    # now shape (Tx, LATENT_DIM_DECODER + LATENT_DIM * 2)
    x = attn_concat_layer([h, st_1])

    # neural net first layer
    x = attn_dense1(x)

    # neural net second layer with special softmax over time
    alphas = attn_dense2(x)

    # Dot the alphas and the h's
    # a.dot(b) = sum over a[t] * b[t]
    context = attn_dot([alphas, h])
    
    return context

# define the rest of the decoder (setelah attention)
decoder_lstm = LSTM(LATENT_DIM_DECODER, return_state=True)
decoder_dense = Dense(num_words_outputs, activation='softmax')

initial_s = Input(shape=(LATENT_DIM_DECODER,), name='s0')
initial_c = Input(shape=(LATENT_DIM_DECODER,), name='c0')
context_last_word_concat_layer = Concatenate(axis=2)

# unlike seq2seq, we cannot get the output all in one step
# instead we need to do Ty steps
# and in each of those steps, we need to consider all Tx h's

s = initial_s
c = initial_c

outputs = []
for t in range(max_len_target):
    # get context using attention
    context = one_step_attention(encoder_outputs, s)

    # we need a different layer for each time step
    selector = Lambda(lambda x: x[:, t:t+1])
    xt = selector(decoder_inputs_x)

    # combine
    decoder_lstm_input = context_last_word_concat_layer([context, xt])

    # pass the combined [context, last word] into the LSTM along with [s, c]
    # get the new [s, c] and output
    o, s, c = decoder_lstm(decoder_lstm_input, initial_state=[s, c])

    # final dense layer to get next word prediction
    decoder_outputs = decoder_dense(o)
    outputs.append(decoder_outputs)

# outputs is now a list of length Ty
# each element os of shape (batch size, output vocab size)
# therefore id we simply stack all the outputs into 1 tensor it would be of shape T x N x D
# we would like it to be of shape N x T x D

def stack_and_transpose(x):
    # x is a list of length T, each element is a batch_size x output_vocab_size tensor
    x = K.stack(x) # is now T x batch_size x output_vocab_size tensor
    x = K.permute_dimensions(x, pattern=(1, 0, 2)) # is now batch_size x T x output_vocab_size tensor
    return x

# make it a layer
stacker = Lambda(stack_and_transpose)
outputs = stacker(outputs)

# create model
model = Model(
    inputs=[
        encoder_inputs_placeholder,
        decoder_inputs_placeholder,
        initial_s,
        initial_c
    ],
    outputs=outputs
)

# compile the model
model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# train the model
z = np.zeros((NUM_SAMPLES, LATENT_DIM_DECODER)) # initial [s, c]
r = model.fit(
    [encoder_inputs, decoder_inputs, z, z], decoder_targets_one_hot,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.2
)

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(r.history['accuracy'], label='accuracy')
plt.plot(r.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()

# save model
model.save(current_dir + '/model/malang/pre_trained_base_model.keras')

# MAKE PREDICTIONS
# we need to create another model that can take in the RNN state and previous word as input
# and accept a T=1 sequence

# the encoder will be stand-alone
# from this we will get out initial decoder hidden states
# i.e. h(1), ..., h(Tx)
encoder_model = Model(encoder_inputs_placeholder, encoder_outputs)

# next we define a T=1 decoder model
encoder_outputs_as_input = Input(shape=(max_len_input, LATENT_DIM * 2,))
decoder_inputs_single = Input(shape=(1,))
decoder_inputs_single_x = decoder_embedding(decoder_inputs_single)

# no need to loop over attention steps this time because there is only one step
context = one_step_attention(encoder_outputs_as_input, initial_s)

# combine context with last word
decoder_lstm_input = context_last_word_concat_layer([context, decoder_inputs_single_x])

# lstm and final dense
o, s, c = decoder_lstm(decoder_lstm_input, initial_state=[initial_s, initial_c])
decoder_outputs = decoder_dense(o)

# note we dont really need the final stack and transpose because there's only 1 output
# it is already of size N x D no need to make it 1 x N x D -> N x 1 x D

# create the model object
decoder_model = Model(
    inputs=[
        decoder_inputs_single,
        encoder_outputs_as_input,
        initial_s,
        initial_c
    ],
    outputs=[decoder_outputs, s, c]
)

# kembalikan ke words
idx2word_asli = {v:k for k, v in word2idx_inputs.items()}
idx2word_trans = {v:k for k, v in word2idx_outputs.items()}

# save model
encoder_model.save(current_dir + '/model/malang/pre_trained_encoder_model.keras')
decoder_model.save(current_dir + '/model/malang/pre_trained_decoder_model.keras')

# save config
with open(current_dir + '/model/malang/train_config.json', 'w', encoding='utf-8') as f:
    array_data = {
        "max_len_input": max_len_input,
        "max_len_target": max_len_target,
        "word2idx_inputs": word2idx_inputs,
        "word2idx_outputs": word2idx_outputs, 
        "num_words_outputs": num_words_outputs, 
        "num_words": num_words,
        "idx2word_trans": idx2word_trans,
        "input_texts": input_texts,
        "target_texts": target_texts,
        "LATENT_DIM_DECODER": LATENT_DIM_DECODER,
    }

    json.dump(array_data, f)

print('train selesai')
exit()
def decode_sequence(input_seq):
    # encode the input as state vectors
    enc_out = encoder_model.predict(input_seq)

    # generate empty target sequencfe of length 1
    target_seq = np.zeros((1,1))

    # populate the first character of target sequence with the start character
    target_seq[0,0] = word2idx_outputs['<sos>']

    eos = word2idx_outputs['<eos>']

    # [s,c] will be updated in each loop iter
    s = np.zeros((1, LATENT_DIM_DECODER))
    c = np.zeros((1, LATENT_DIM_DECODER))

    # create the translation
    output_sentence = []
    for _ in range(max_len_target):
        o, s, c = decoder_model.predict([target_seq, enc_out, s, c])

        # get next word
        idx = np.argmax(o.flatten())

        if eos == idx:
            break

        word = ''
        if idx > 0:
            word = idx2word_trans[idx]
            output_sentence.append(word)

        # update the decoder input, which is the word just generated
        target_seq[0,0] = idx
    
    return ' '.join(output_sentence)

while True:
    i = np.random.choice(len(input_texts))
    input_seq = encoder_inputs[i:i+1]
    translation = decode_sequence(input_seq)
    print('input:', input_texts[i])
    print('predicted:', translation)
    print('actual:', target_texts[i])
    
    ans = input('-----continue? [Y/n]-----')
    if ans and ans[0].lower().startswith('n'):
        break