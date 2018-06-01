import time
import math
import torch

USE_CUDA = torch.cuda.is_available()
if USE_CUDA: print("Using CUDA")


mcell = "lstm" # memory cell


def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        try:
            tensor[c] = all_characters.index(string[c])
        except:
            continue
    return tensor

# Readable time elapsed

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


#%%

def read_file( filename = "poets.txt" , char_filename = "unique_chars.txt" ):
    file = open(filename, "r", encoding = "utf-8" ).read() 
    # remember, we should not use "strip" because we need the newline character "\n" as EOS.

    all_chars = open( char_filename, "r", encoding = "utf-8" ).read()
    all_characters = [ _ for _ in all_chars ] # break string into array
    
    n_characters = len( all_characters )

    return file, len(file), all_characters, n_characters


file, file_len, all_characters, n_characters = read_file( filename = "poets.txt" )



