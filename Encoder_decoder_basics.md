## Transformer_end_to_end
### Basics of Transformers 
 [Introduction](##Introduction)
 
 ##Introduction
 In this we will be covering basics of Transformers Architecture and Implementations of required 
 components.As we seen “Attention is all you need” changed the 
 ways of providing strength to NLP complex problems.
 
 As RNN is taking sequences and thus needs previous hidden outputs, we goes 
 on stacking we can able to get better results but as we goes on 
 increasing size of inputs LSTM came with Gates mechanism addressed in
 long term dependency problem. 
 
 Transformers came out to be very ideal level capabilities with 
 Encoder and Decoder Implementations.So dealing with such Architecture
 Tensorflow and Pytorch became user's likely framework.
 
 ## Encoder Decoder 
 
 Before diving into Transformers it is good to have clarity with Encoder and Decoder Implementations.
 
 A basic Approach 
 ref https://edumunozsala.github.io/BlogEms/fastpages/jupyter/encoder-decoder/lstm/attention/tensorflow%202/2020/10/07/Intro-seq2seq-Encoder-Decoder-ENG-SPA-translator-tf2.html#Intro-to-the-Encoder-Decoder-model-and-the-Attention-mechanism
  
 ![Alt text](./images/encoder_decoder_basic1.png?raw=true "cummalive plot")
 
 ### Steps to follow
 Importing the libraries and initialize global variables
 ```
import os
import gc
import time
import re
import unicodedata

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```

```
#Importing libraries
import tensorflow as tf
from tensorflow.keras import backend as K

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
```

Setting the variables for Data Location 
```
# Global parameters
#root folder
root_folder='.'
#data_folder='.'
data_folder_name='data'
train_filename='spa.txt'

# Variable for data directory
DATA_PATH = os.path.abspath(os.path.join(root_folder, data_folder_name))
train_filenamepath = os.path.abspath(os.path.join(DATA_PATH, train_filename))

# Both train and test set are in the root data directory
train_path = DATA_PATH
test_path = DATA_PATH
```

Defining parameters and Hyperparameters
```
# Parameters for our model
INPUT_COLUMN = 'input'
TARGET_COLUMN = 'target'
TARGET_FOR_INPUT = 'target_for_input'
NUM_SAMPLES = 20000 #40000
MAX_VOCAB_SIZE = 20000
EMBEDDING_DIM = 128
HIDDEN_DIM=1024 #512

BATCH_SIZE = 64  # Batch size for training.
EPOCHS = 10  # Number of epochs to train for.

ATTENTION_FUNC='general'
```

Getting the Data 
http://www.manythings.org/anki/

Processing the Text data
```
# Some function to preprocess the text data, taken from the Neural machine translation with attention tutorial
# in Tensorflow
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
    ''' Preprocess the input text w applying lowercase, removing accents, 
    creating a space between a word and the punctuation following it and 
    replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    Input:
        - w: a string, input text
    Output:
        - a string, the cleaned text
    '''
    w = unicode_to_ascii(w.lower().strip())

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    #w = '<start> ' + w + ' <end>'
    
    return w
```


As we see alots of steps is to be carried out 




## References 
http://nlp.seas.harvard.edu/2018/04/03/attention.html
https://bastings.github.io/annotated_encoder_decoder/
http://www.adeveloperdiary.com/data-science/deep-learning/nlp/machine-translation-using-attention-with-pytorch/#types-of-attention
https://theaisummer.com/transformer/
