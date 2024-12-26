import os
import json
import numpy as np
from keras.models import load_model
from tensorflow.keras.utils import pad_sequences
import keras.backend as K

def softmax_over_time(x):
    assert(K.ndim(x) > 2)
    e = K.exp(x - K.max(x, axis=1, keepdims=True))
    s = K.sum(e, axis=1, keepdims=True)
    return e/s

current_dir = os.getcwd()

config = []
with open(current_dir + '/model/malang/train_config.json', encoding='utf-8') as f:
    config = json.load(f)

max_len_input = config['max_len_input']
max_len_target = config['max_len_target']
word2idx_inputs = config['word2idx_inputs']
word2idx_outputs = config['word2idx_outputs']
num_words_outputs = config['num_words_outputs']
num_words = config['num_words']
idx2word_trans = config['idx2word_trans']
input_texts = config['input_texts']
target_texts = config['target_texts']
LATENT_DIM_DECODER = config['LATENT_DIM_DECODER']

encoder_model = load_model(current_dir + '/model/malang/pre_trained_encoder_model.keras')
decoder_model = load_model(current_dir + '/model/malang/pre_trained_decoder_model.keras', custom_objects={
    'softmax_over_time': softmax_over_time
})

def decode_sequence(message):
    encoded = [word2idx_inputs[s] if s in word2idx_inputs else 0 for s in message.strip().lower().split()]
    input_seq = pad_sequences([encoded], maxlen=max_len_input)
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
            word = idx2word_trans[str(idx)]
            output_sentence.append(word)

        # update the decoder input, which is the word just generated
        target_seq[0,0] = idx
    
    return ' '.join(output_sentence)

if __name__ == '__main__':
    while True:
        message = input('Kamu: ')
        if message == 'quit':
            break
        
        translation = decode_sequence(message)
        print('Bot : ' + translation)