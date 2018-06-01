#!/usr/bin/env python
# https://github.com/spro/char-rnn.pytorch

import torch
from torch.autograd import Variable

import helpers


def generate(decoder, prime_str='我', predict_len=100, temperature=0.8, cuda=False):
    hidden = decoder.init_hidden(1)
    prime_input = Variable( helpers.char_tensor(prime_str).unsqueeze(0))

    if helpers.USE_CUDA:
        if helpers.mcell== "lstm": hidden = (hidden[0].cuda(), hidden[1].cuda())
        else: hidden = hidden.cuda()
        prime_input = prime_input.cuda()
    predicted = prime_str

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = decoder(prime_input[:,p], hidden)
        
    inp = prime_input[:,-1]
    
    for p in range(predict_len):
        output, hidden = decoder(inp, hidden)
        
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
#        print( top_i )
        predicted_char = helpers.all_characters[top_i]
        if predicted_char == "\n": break # end of sentence
        predicted += predicted_char
        inp = Variable( helpers.char_tensor(predicted_char).unsqueeze(0) )
        if cuda:
            inp = inp.cuda()

    return predicted.strip()


def generate_random_str( prime_str = "新年", predict_len = 40 , model = None):
    model_filename = "poets.pt"
    tempeature = 0.8
    
    if model is None:
        model = torch.load( model_filename )
    
    rst = generate( model, prime_str, predict_len , tempeature, helpers.USE_CUDA )

    print( rst )


def generate_cts( prime_str = "儿童节快乐", predict_len = 40, sentence_len = 6 , model = None):
    model_filename = "poets.pt"
    tempeature = 0.8
    
    if model is None:
        model = torch.load( model_filename )
    
    poet = []
    
    for s in prime_str:
        rst = ""
#        while len(rst) != sentence_len:
        while len(rst) <= sentence_len:
            rst = generate( model, s, predict_len , tempeature, helpers.USE_CUDA )
        poet.append( rst[:sentence_len] )
        print( poet[-1] )
    
    print( poet )

if __name__ == '__main__':    
#    generate_random_str()
    generate_cts()
