# TODO

1. Create a "main.py" program to run the program from beginning to end.
    - if poets.txt doesn't exist, then run preprocess first
    - if poets.pt doesn't exist, then run train first
    - run generate
    
    
2. Create an argument in helper to control either use "GRU" or "LSTM"
    - Change model.py accordingly. (hint: about 2 - 5 lines change in model.py)
    
    
3. Train.py should read each line of poet independently.


4. Change generate.py so that it can generate 4-line poet with same length.



# Reference
[PyTorch RNN Tutorial](http://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html)
[char-rnn](https://github.com/spro/char-rnn.pytorch)


