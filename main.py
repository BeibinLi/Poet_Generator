import os
from preprocess import xml2txt
from train import train
from generate import generate_random_str

if not os.path.exists( "poets.txt" ): 
    xml2txt()

if not os.path.exists( "poets.pt" ):  
    train()


generate_random_str()



