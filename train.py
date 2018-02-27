

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

def random_training_set(file, chunk_len, batch_size):
    inp = torch.LongTensor(batch_size, chunk_len)
    target = torch.LongTensor(batch_size, chunk_len)
    
    file_len = len( file )
    for bi in range(batch_size):
        start_index = random.randint(0, file_len - chunk_len)
        end_index = start_index + chunk_len + 1
        chunk = file[start_index:end_index]
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
    if helpers.USE_CUDA: hidden = hidden.cuda()
    decoder.zero_grad()
    loss = 0

    for c in range( chunk_len ):
        output, hidden = decoder(inp[:,c], hidden)
        loss += criterion(output.view(batch_size, -1), target[:,c])

    loss.backward()
    decoder_optimizer.step()

    return loss.data[0] / chunk_len

def train( filename = "poets.txt", hidden_size = 100, n_layers = 2, 
          learning_rate=0.01, n_epochs = 2000, chunk_len=200, batch_size = 128,
          shuffle = True, print_every =  100 ):
    #%% Global Configuration
    file, file_len, all_characters, n_characters = helpers.read_file( filename )
    
    print( "There are %d in the dataset" % n_characters )
    
    #%% Model Saving and Loading
    
    model_filename = os.path.splitext(os.path.basename(filename))[0] + '.pt'

    if os.path.exists( model_filename ):
        decoder = load( model_filename )
    else:
        decoder = CharRNN(
            n_characters,
            hidden_size,
            n_characters,
            model="gru",
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
            inp, target = random_training_set( file, chunk_len, batch_size )
            
            loss = train_one_entry(decoder, decoder_optimizer, criterion, 
                                        inp, target, chunk_len, batch_size )
            
            all_losses.append( loss )
    
            if epoch % print_every == 0:
                print('[%s (%d %d%%) %.4f]' % ( helpers.time_since(start), epoch, epoch / n_epochs * 100, loss))
                print(generate(decoder, '新年', 40, cuda= helpers.USE_CUDA), '\n')
                    
                save( decoder, model_filename )
    
    except KeyboardInterrupt:
        save( decoder, model_filename )
        
        
    import matplotlib.pyplot as plt
    plt.plot( all_losses )
    plt.xlabel( "iteration" )
    plt.ylabel( "train lost" )

if __name__ == "__main__":
    train()
    
    
