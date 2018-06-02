

import torch
import torch.nn as nn
from torch.autograd import Variable
import os, time, random
import helpers
from model import CharRNN
from generate import generate


def save(decoder, model_filename):
    torch.save(decoder, model_filename)
    print('Saved as %s' % model_filename)


def load( model_filename ):
    net = torch.load( model_filename )
    print("Model loaded from", model_filename )
    return net


#%%

def random_training_set(sentences, max_len, batch_size):
# Output:inp (size chunk_len x batch_size ), target (same size, shift one column)
    inp = torch.LongTensor(batch_size, max_len)
    target = torch.LongTensor(batch_size, max_len)
    
    for bi in range(batch_size):
        
        chunk = random.choice( sentences )
        
        if len(chunk) > max_len + 1: # remove extra
            chunk = chunk[:max_len+1]
        else: # add padding
            chunk += "\n" * (max_len + 1 - len(chunk))        
        inp[bi] = helpers.char_tensor(chunk[:-1])
        target[bi] = helpers.char_tensor(chunk[1:])
    inp = Variable(inp)
    target = Variable(target)
    if helpers.USE_CUDA:
        inp = inp.cuda()
        target = target.cuda()
    return inp, target

def train_one_entry(decoder, decoder_optimizer, criterion, inp, target, chunk_len, batch_size ):
    hidden = decoder.init_hidden(batch_size)
    
    if helpers.USE_CUDA:
        if helpers.mcell == "lstm": hidden = (hidden[0].cuda(), hidden[1].cuda())
        else: hidden = hidden.cuda()
    decoder.zero_grad()
    loss = 0

    for c in range( chunk_len ):
        output, hidden = decoder(inp[:,c], hidden)
        loss += criterion(output.view(batch_size, -1), target[:,c])

    loss.backward()
    decoder_optimizer.step()

    return loss.data[0] / chunk_len

def train( filename = "poets.txt", hidden_size = 128, n_layers = 2, 
          learning_rate=0.01, n_epochs = 10000, chunk_len=20, batch_size = 1024,
          print_every =  100 ):
    #%% Global Configuration
    file, file_len, all_characters, n_characters = helpers.read_file( filename )
    
    sentences = file.split("\n")
    
    print( "There are %d unique characters in the dataset" % n_characters )
    print( "There are %d sentences in the dataset with total of %d characters" % ( len(sentences), len(file) ) )
    
    #%% Model Saving and Loading
    model_filename = helpers.pt_name

    if os.path.exists( model_filename ):
        decoder = load( model_filename )
    else:
        decoder = CharRNN(
            n_characters,
            hidden_size,
            n_characters,
            model = helpers.mcell,
            n_layers=n_layers,
        )
        
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    if helpers.USE_CUDA: decoder.cuda()
    
    start = time.time()
    all_losses = []
    
    try:
        print("Training for %d epochs..." % n_epochs)
        for epoch in range(n_epochs):
            
            if epoch != 0 and epoch % 1000 == 0: 
                learning_rate /= 2
                decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)

            inp, target = random_training_set( sentences, chunk_len, batch_size )
            
            loss = train_one_entry(decoder, decoder_optimizer, criterion, 
                                        inp, target, chunk_len, batch_size )
            
            all_losses.append( loss )
    
            if epoch != 0 and epoch % print_every == 0:
                print('%s: [%s (%d %d%%) %.4f]' % ( time.ctime(), helpers.time_since(start), epoch, epoch / n_epochs * 100, loss))
                print(generate(decoder, '新年', 100, cuda= helpers.USE_CUDA), '\n')
                    
                save( decoder, model_filename )
    
    except KeyboardInterrupt:
        save( decoder, model_filename )
        
        
    import matplotlib.pyplot as plt
    plt.plot( all_losses )
    plt.xlabel( "iteration" )
    plt.ylabel( "train loss" )

if __name__ == "__main__":
    train()
    
    
