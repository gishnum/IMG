#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 20:14:15 2019

@author: gishnu
"""

import os
from pycocotools.coco import COCO

# initialize COCO API for instance annotations
dataDir = '.'
dataType = 'val2014'
instances_annFile = os.path.join(dataDir, 'cocoapi/annotations/instances_{}.json'.format(dataType))
coco = COCO(instances_annFile)

# initialize COCO API for caption annotations
captions_annFile = os.path.join(dataDir, 'cocoapi/annotations/captions_{}.json'.format(dataType))
coco_caps = COCO(captions_annFile)

# get image ids 
ids = list(coco.anns.keys())


import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt


# pick a random image and obtain the corresponding URL
ann_id = np.random.choice(ids)
img_id = coco.anns[ann_id]['image_id']
img = coco.loadImgs(img_id)[0]
url = img['coco_url']

# print URL and visualize corresponding image
print(url)
I = io.imread(url)
plt.axis('off')
plt.imshow(I)
plt.show()

# load and display captions
annIds = coco_caps.getAnnIds(imgIds=img['id']);
anns = coco_caps.loadAnns(annIds)
coco_caps.showAnns(anns)




#!/usr/bin/env python
# coding: utf-8

# # Image Captioning
# 
# ## Part 1: Load and Pre-Process Data and Experiment with Models
# 
# ---
# 
# In this notebook, we will learn how to load and pre-process data from the [COCO dataset](http://cocodataset.org/#home). We will also experiment with a CNN-RNN model for automatically generating image captions. These are *not* the final models that we will use. For the final ones, see **model.py**.
# 
# Use the links below to navigate the notebook:
# - [Step 1](#step1): Explore the Data Loader
# - [Step 2](#step2): Use the Data Loader to Obtain Batches
# - [Step 3](#step3): Experiment with the CNN Encoder
# - [Step 4](#step4): Implement the RNN Decoder

# <a id='step1'></a>
# ## Step 1: Explore the Data Loader
# 
# We will use a [data loader](http://pytorch.org/docs/master/data.html#torch.utils.data.DataLoader) to load the COCO dataset in batches. 
# 
# In the code cell below, we will initialize the data loader by using the `get_loader` function in **data_loader.py**.  
# 
# The `get_loader` function takes as input a number of arguments that can be explored in **data_loader.py**. Most of the arguments must be left at their default values; we may amend the values of the arguments below:
# 1. **`transform`** - an [image transform](http://pytorch.org/docs/master/torchvision/transforms.html) specifying how to pre-process the images and convert them to PyTorch tensors before using them as input to the CNN encoder.
# 2. **`mode`** - one of `'train'`, `'val'` (loads the training or validation data in batches) or `'test'` (for the test data). We will say that the data loader is in training, validation or test mode, respectively.
# 3. **`batch_size`** - determines the batch size.  When training/validating the model, this is number of image-caption pairs used to amend the model weights in each training/validation step.
# 4. **`vocab_threshold`** - the total number of times that a word must appear in the training captions before it is used as part of the vocabulary.  Words that have fewer than `vocab_threshold` occurrences in the training captions are considered unknown words. 
# 5. **`vocab_from_file`** - a Boolean that decides whether to load the vocabulary from file.  
# 
# We will describe the `vocab_threshold` and `vocab_from_file` arguments in more detail soon.

# In[1]:


# Watch for any changes in vocabulary.py, data_loader.py or model.py, and re-load it automatically.
#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import torch
from data_loader import get_loader
from torchvision import transforms

# Define a transform to pre-process the training images.
transform_train = transforms.Compose([ 
    transforms.Resize(256),                          # smaller edge of image resized to 256
    transforms.RandomCrop(224),                      # get 224x224 crop from random location
    transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
    transforms.ToTensor(),                           # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))])

# Set the minimum word count threshold.
vocab_threshold = 5

# Specify the batch size.
batch_size = 10

# Obtain the data loader.
data_loader = get_loader(transform=transform_train,
                         mode='train',
                         batch_size=batch_size,
                         vocab_threshold=vocab_threshold,
                         vocab_from_file=False)


# When we ran the code cell above, the data loader was stored in the variable `data_loader`.  
# 
# We can access the corresponding dataset as `data_loader.dataset`.  This dataset is an instance of the `CoCoDataset` class in **data_loader.py**.  For more information on data loaders and datasets see [this PyTorch tutorial](http://pytorch.org/tutorials/beginner/data_loading_tutorial.html).
# 
# ### Exploring the `__getitem__` Method
# 
# The `__getitem__` method in the `CoCoDataset` class determines how an image-caption pair is pre-processed before being incorporated into a batch.  When the data loader is in training or validation mode, this method begins by first obtaining the filename (`path`) of an image and its corresponding caption (`caption`).
# 
# #### Image Pre-Processing 
# 
# Image pre-processing is relatively straightforward (from the `__getitem__` method in the `CoCoDataset` class):
# ```python
# # Convert image to tensor and pre-process using transform
# image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
# image = self.transform(image)
# ```
# After loading the image in the folder with name `path`, the image is pre-processed using the same transform (`transform_train`) that was supplied when instantiating the data loader.  
# 
# #### Caption Pre-Processing 
# 
# The captions also need to be pre-processed and prepped for training. In this example, for generating captions, we are aiming to create a model that predicts the next token of a sentence from previous tokens, so we turn the caption associated with any image into a list of tokenized words, before casting it to a PyTorch tensor that we can use to train the network.
# 
# To understand in more detail how COCO captions are pre-processed, we'll first need to take a look at the `vocab` instance variable of the `CoCoDataset` class.  The code snippet below is pulled from the `__init__` method of the `CoCoDataset` class:
# ```python
# def __init__(self, transform, mode, batch_size, vocab_threshold, vocab_file, start_word, 
#         end_word, unk_word, annotations_file, vocab_from_file, img_folder):
#         ...
#         self.vocab = Vocabulary(vocab_threshold, vocab_file, start_word,
#             end_word, unk_word, annotations_file, vocab_from_file)
#         ...
# ```
# `data_loader.dataset.vocab` is an instance of the `Vocabulary` class from **vocabulary.py**.  
# 
# We use this instance to pre-process the COCO captions (from the `__getitem__` method in the `CoCoDataset` class):
# 
# ```python
# # Convert caption to tensor of word ids.
# tokens = nltk.tokenize.word_tokenize(str(caption).lower())   # line 1
# caption = []                                                 # line 2
# caption.append(self.vocab(self.vocab.start_word))            # line 3
# caption.extend([self.vocab(token) for token in tokens])      # line 4
# caption.append(self.vocab(self.vocab.end_word))              # line 5
# caption = torch.Tensor(caption).long()                       # line 6
# ```
# 
# As we will see soon, this code converts any string-valued caption to a list of integers, before casting it to a PyTorch tensor.  To see how this code works, we'll apply it to the sample caption in the next code cell.

# In[3]:


sample_caption = 'A person doing a trick on a rail while riding a skateboard.'


# In **`line 1`** of the code snippet, every letter in the caption is converted to lowercase, and the [`nltk.tokenize.word_tokenize`](http://www.nltk.org/) function is used to obtain a list of string-valued tokens.

# In[4]:


import nltk

sample_tokens = nltk.tokenize.word_tokenize(str(sample_caption).lower())
print(sample_tokens)


# In **`line 2`** and **`line 3`** we initialize an empty list and append an integer to mark the start of a caption.  This [paper](https://arxiv.org/pdf/1411.4555.pdf) uses a special start word (and a special end word, which we'll examine below) to mark the beginning (and end) of a caption.
# 
# This special start word (`"<start>"`) is decided when instantiating the data loader and is passed as a parameter (`start_word`).  We will keep this parameter at its default value (`start_word="<start>"`).
# 
# As we will see below, the integer `0` is always used to mark the start of a caption.

# In[5]:


sample_caption = []

start_word = data_loader.dataset.vocab.start_word
print('Special start word:', start_word)
sample_caption.append(data_loader.dataset.vocab(start_word))
print(sample_caption)


# In **`line 4`**, we continue the list by adding integers that correspond to each of the tokens in the caption.

# In[6]:


sample_caption.extend([data_loader.dataset.vocab(token) for token in sample_tokens])
print(sample_caption)


# In **`line 5`**, we append a final integer to mark the end of the caption.  
# 
# Identical to the case of the special start word (above), the special end word (`"<end>"`) is decided when instantiating the data loader and is passed as a parameter (`end_word`).  We keep this parameter at its default value (`end_word="<end>"`).
# 
# As we will see below, the integer `1` is always used to  mark the end of a caption.

# In[7]:


end_word = data_loader.dataset.vocab.end_word
print('Special end word:', end_word)

sample_caption.append(data_loader.dataset.vocab(end_word))
print(sample_caption)


# Finally, in **`line 6`**, we convert the list of integers to a PyTorch tensor and cast it to [long type](http://pytorch.org/docs/master/tensors.html#torch.Tensor.long).  More information about the different types of PyTorch tensors is available on the [website](http://pytorch.org/docs/master/tensors.html).

# In[8]:


sample_caption = torch.Tensor(sample_caption).long()
print(sample_caption)


# And that's it!  In summary, any caption is converted to a list of tokens, with _special_ start and end tokens marking the beginning and end of the sentence:
# ```
# [<start>, 'a', 'person', 'doing', 'a', 'trick', 'while', 'riding', 'a', 'skateboard', '.', <end>]
# ```
# This list of tokens is then turned into a list of integers, where every distinct word in the vocabulary has an associated integer value:
# ```
# [0, 3, 98, 754, 3, 396, 207, 139, 3, 753, 18, 1]
# ```
# Finally, this list is converted to a PyTorch tensor.  All of the captions in the COCO dataset are pre-processed using this same procedure from **`lines 1-6`** described above.  
# 
# As we saw, in order to convert a token to its corresponding integer, we call `data_loader.dataset.vocab` as a function.  The details of how this call works can be explored in the `__call__` method in the `Vocabulary` class in **vocabulary.py**.  
# 
# ```python
# def __call__(self, word):
#     if not word in self.word2idx:
#         return self.word2idx[self.unk_word]
#     return self.word2idx[word]
# ```
# 
# The `word2idx` instance variable is a Python dictionary that is indexed by string-valued keys (mostly tokens obtained from training captions).  For each key, the corresponding value is the integer that the token is mapped to in the pre-processing step.
# 
# Use the code cell below to view a subset of this dictionary.  We also print the total number of keys.

# In[9]:


# Preview the word2idx dictionary.
print (dict(list(data_loader.dataset.vocab.word2idx.items())[:10]))

# Print the total number of keys in the word2idx dictionary.
print('Total number of tokens in vocabulary:', len(data_loader.dataset.vocab))


# In **vocabulary.py**, the `word2idx` dictionary is created by looping over the captions in the training dataset.  If a token appears no less than `vocab_threshold` times in the training set, then it is added as a key to the dictionary and assigned a corresponding unique integer.  In general, **smaller** values for `vocab_threshold` yield a **larger** number of tokens in the vocabulary.  We can see this in the next two code cells.

# In[10]:


# Minimum word count threshold.
vocab_threshold = 5

# Obtain the data loader.
data_loader = get_loader(transform=transform_train,
                         mode='train',
                         batch_size=batch_size,
                         vocab_threshold=vocab_threshold,
                         vocab_from_file=False)
# Print the total number of keys in the word2idx dictionary.
print('Total number of tokens in vocabulary:', len(data_loader.dataset.vocab))


# In[11]:


# Minimum word count threshold.
vocab_threshold = 10

# Obtain the data loader.
data_loader = get_loader(transform=transform_train,
                         mode='train',
                         batch_size=batch_size,
                         vocab_threshold=vocab_threshold,
                         vocab_from_file=False)

# Print the total number of keys in the word2idx dictionary.
print('Total number of tokens in vocabulary:', len(data_loader.dataset.vocab))


# There are also a few special keys in the `word2idx` dictionary.  Other than the special start word (`"<start>"`) and special end word (`"<end>"`), there is one more special token, corresponding to unknown words (`"<unk>"`).  All tokens that don't appear anywhere in the `word2idx` dictionary are considered unknown words.  In the pre-processing step, any unknown tokens are mapped to the integer `2`.

# In[12]:


unk_word = data_loader.dataset.vocab.unk_word
print('Special unknown word:', unk_word)

print('All unknown words are mapped to this integer:', data_loader.dataset.vocab(unk_word))
print ("For example:")
print("'jfkafejw' is mapped to", data_loader.dataset.vocab('jfkafejw'))


# The final thing to mention is the `vocab_from_file` argument that is supplied when creating a data loader.  When we create a new data loader, the vocabulary (`data_loader.dataset.vocab`) is saved as a [pickle](https://docs.python.org/3/library/pickle.html) file in the project folder, with filename `vocab.pkl`.
# 
# If we are still tweaking the value of the `vocab_threshold` argument, we **must** set `vocab_from_file=False` to have our changes take effect.  
# 
# But once we are happy with the value that we have chosen for the `vocab_threshold` argument, we need only run the data loader *one more time* with our chosen `vocab_threshold` to save the new vocabulary to file.  Then, we can henceforth set `vocab_from_file=True` to load the vocabulary from file and speed the instantiation of the data loader.  Note that building the vocabulary from scratch is the most time-consuming part of instantiating the data loader, and so we are strongly encouraged to set `vocab_from_file=True` as soon as we are able.
# 
# Note that if `vocab_from_file=True`, then any supplied argument for `vocab_threshold` when instantiating the data loader is completely ignored.

# In[13]:


# Obtain the data loader (from file). Note that it runs much faster than before!
data_loader = get_loader(transform=transform_train,
                         mode='train',
                         batch_size=batch_size,
                         vocab_from_file=True)


# <a id='step2'></a>
# ## Step 2: Use the Data Loader to Obtain Batches
# 
# The captions in the dataset vary greatly in length.  We can see this by examining `data_loader.dataset.caption_lengths`, a Python list with one entry for each training caption (where the value stores the length of the corresponding caption).  
# 
# In the code cell below, we use this list to print the total number of captions in the training data with each length.  As we will see below, the majority of captions have length 10.  Likewise, very short and very long captions are quite rare.  

# In[14]:


from collections import Counter

# Tally the total number of training captions with each length.
counter = Counter(data_loader.dataset.caption_lengths)
lengths = sorted(counter.items(), key=lambda pair: pair[1], reverse=True)
for value, count in lengths:
    print('value: %2d --- count: %5d' % (value, count))


# To generate batches of training data, we begin by first sampling a caption length (where _the probability that any length is drawn is proportional to the number of captions with that length_ in the dataset).  Then, we retrieve a batch of size `batch_size` of image-caption pairs, where all captions have the sampled length.  This approach for assembling batches matches the procedure in [this paper](https://arxiv.org/pdf/1502.03044.pdf) and has been shown to be computationally efficient without degrading performance.
# 
# Run the code cell below to generate a batch.  The `get_indices` method in the `CoCoDataset` class first samples a caption length, and then samples `batch_size` indices corresponding to training data points with captions of that length.  These indices are stored below in `indices`.
# 
# These indices are supplied to the data loader, which then is used to retrieve the corresponding data points.  The pre-processed images and captions in the batch are stored in `images` and `captions`.

# In[15]:


import numpy as np
import torch.utils.data as data

# Randomly sample a caption length, and sample indices with that length.
indices = data_loader.dataset.get_indices()
print('{} sampled indices: {}'.format(len(indices), indices))
# Create and assign a batch sampler to retrieve a batch with the sampled indices.
new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
data_loader.batch_sampler.sampler = new_sampler

# Obtain the batch.
for batch in data_loader:
    images, captions = batch[0], batch[1]
    break
    
print('images.shape:', images.shape)
print('captions.shape:', captions.shape)

# Print the pre-processed images and captions.
#print('images:', images)
#print('captions:', captions)


# <a id='step3'></a>
# ## Step 3: Experiment with the CNN Encoder
# 
# First, we will import `EncoderCNN` and `DecoderRNN` from **model.py**. 

# In[16]:


# Import EncoderCNN and DecoderRNN. 
from model import EncoderCNN, DecoderRNN


# Now we will instantiate the CNN encoder in `encoder`.  
# 
# The pre-processed images from the batch in **Step 2** of this notebook are then passed through the encoder, and the output is stored in `features`. The assert statement ensures that `features` has shape `[batch_size, embed_size]`.

# In[17]:


# Specify the dimensionality of the image embedding.
embed_size = 256

# Initialize the encoder. (We can add additional arguments if necessary.)
encoder = EncoderCNN(embed_size)

# Move the encoder to GPU if CUDA is available.
if torch.cuda.is_available():
    encoder = encoder.cuda()
    
# Move the last batch of images from Step 2 to GPU if CUDA is available
if torch.cuda.is_available():
    images = images.cuda()
# Pass the images through the encoder.
features = encoder(images)

print('type(features):', type(features))
print('features.shape:', features.shape)

# Check that our encoder satisfies some requirements of the project!
assert (features.shape[0]==batch_size) & (features.shape[1]==embed_size), "The shape of the encoder output is incorrect."


# This encoder uses the pre-trained ResNet-50 architecture (with the final fully-connected layer removed) to extract features from a batch of pre-processed images.  The output is then flattened to a vector, before being passed through a `Linear` layer to transform the feature vector to have the same size as the word embedding.
# 
# ![Encoder](images/encoder.png)
# 
# We could amend the encoder in **model.py**, to experiment with other architectures, such as using a [different pre-trained model architecture](http://pytorch.org/docs/master/torchvision/models.html) or [adding batch normalization](http://pytorch.org/docs/master/nn.html#normalization-layers).  
# 
# For this project, we will **incorporate a pre-trained CNN into our encoder**.  The `EncoderCNN` class must take `embed_size` as an input argument, which will also correspond to the dimensionality of the input to the RNN decoder that we will implement in Step 4.  When we train our model in the next notebook in this sequence (**2_Training.ipynb**), we will tweak the value of `embed_size`.

# <a id='step4'></a>
# ## Step 4: Implement the RNN Decoder
# 
# Our decoder will be an instance of the `DecoderRNN` class from **model.py** and must accept as input:
# - the PyTorch tensor `features` containing the embedded image features (outputted in Step 3, when the last batch of images from Step 2 was passed through `encoder`), along with
# - a PyTorch tensor corresponding to the last batch of captions (`captions`) from Step 2.
# 
# Every training batch will contain pre-processed captions where all have the same length (`captions.shape[1]`), so **we won't need to worry about padding**.  
# 
# Although we will test the decoder using the last batch that is currently stored in the notebook, our decoder should accept an arbitrary batch (of embedded image features and pre-processed captions [where all captions have the same length]) as input.  
# 
# ![Decoder](images/decoder.png)
# 
# In the code cell below, `outputs` should have size `[batch_size, captions.shape[1], vocab_size]`.  Our output should be designed such that `outputs[i,j,k]` contains the model's predicted score, indicating how likely the `j`-th token in the `i`-th caption in the batch is the `k`-th token in the vocabulary.  In the next notebook of the sequence (**2_Training.ipynb**), we will supply these scores to the [`torch.nn.CrossEntropyLoss`](http://pytorch.org/docs/master/nn.html#torch.nn.CrossEntropyLoss) optimizer in PyTorch.

# In[18]:


# Specify the number of features in the hidden state of the RNN decoder.
hidden_size = 512

# Store the size of the vocabulary.
vocab_size = len(data_loader.dataset.vocab)

# Initialize the decoder.
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
# Move the decoder to GPU if CUDA is available.
if torch.cuda.is_available():
    decoder = decoder.cuda()
    
# Move the last batch of captions (from Step 1) to GPU if cuda is availble 
if torch.cuda.is_available():
    captions = captions.cuda()
# Pass the encoder output and captions through the decoder
outputs = decoder(features, captions)

print('type(outputs):', type(outputs))
print('outputs.shape:', outputs.shape)

# Check that our decoder satisfies some requirements of the project!
assert (outputs.shape[0]==batch_size) & (outputs.shape[1]==captions.shape[1]) & (outputs.shape[2]==vocab_size), "The shape of the decoder output is incorrect."


############### training ###############

#!/usr/bin/env python
# coding: utf-8

# # Image Captioning
# 
# ## Part 2: Train a CNN-RNN Model
# 
# ---
# 
# In this notebook, we will train our CNN-RNN model.  
# 
# - [Step 1](#step1): Training Setup
#   - [1a](#1a): CNN-RNN architecture
#   - [1b](#1b): Hyperparameters and other variables
#   - [1c](#1c): Image transform
#   - [1d](#1d): Data loader
#   - [1e](#1e): Loss function, learnable parameters and optimizer
# 
# 
# - [Step 2](#step2): Train and Validate the Model
#   - [2a](#2a): Train for the first time
#   - [2b](#2b): Resume training
#   - [2c](#2c): Validation
#   - [2d](#2d): Notes regarding model validation

# <a id='step1'></a>
# ## Step 1: Training Setup
# 
# We will describe the model architecture and specify hyperparameters and set other options that are important to the training procedure. We will refer to [this paper](https://arxiv.org/pdf/1502.03044.pdf) and [this paper](https://arxiv.org/pdf/1411.4555.pdf) for useful guidance.
# 
# <a id='1a'></a>
# ### CNN-RNN architecture
# 
# For the complete CNN-RNN model, see **model.py**. 
# 
# - For the encoder model, we use a pre-trained ResNet which has been known to achieve great success in image classification. We use batch normalization because according to [this paper](https://arxiv.org/abs/1502.03167) it "allows us to use much higher learning rates and be less careful about initialization. It also acts as a regularizer, in some cases eliminating the need for Dropout."
# - The decoder is an RNN which has an Embedding layer, a LSTM layer and a fully-connected layer. LSTM has been shown to be successful in sequence generation.
# 
# <a id='1b'></a>
# ### Hyperparameters and other variables
# 
# In the next code cell, we will set the values for:
# 
# - `batch_size` - the batch size of each training batch.  It is the number of image-caption pairs used to amend the model weights in each training step. We will set it to `32`.
# - `vocab_threshold` - the minimum word count threshold.  A larger threshold will result in a smaller vocabulary, whereas a smaller threshold will include rarer words and result in a larger vocabulary. We will set it to `5` just like [this paper](https://arxiv.org/pdf/1411.4555.pdf)
# - `vocab_from_file` - a Boolean that decides whether to load the vocabulary from file. This will be changed to `True` once we are done setting `vocab_threshold` and generating a `vocab.pkl` file.
# - `embed_size` - the dimensionality of the image and word embeddings. We have tried `512` as done in [this paper](https://arxiv.org/pdf/1411.4555.pdf) but it took a long time to train, so I will set it to `256`.
# - `hidden_size` - the number of features in the hidden state of the RNN decoder. We will use `512` based on [this paper](https://arxiv.org/pdf/1411.4555.pdf). The larger the number, the better the RNN model can memorize sequences. However, larger numbers can significantly slow down the training process.
# - `num_epochs` - the number of epochs to train the model.  We are dealing with a huge amount of data so it will take a long time to complete even 1 epoch. Therefore, we will set `num_epochs` to `1`. We will save the model AND the optimizer every 100 training steps, and to resume training from the last step.

# In[1]:


# Watch for any changes in vocabulary.py, data_loader.py, utils.py or model.py, and re-load it automatically.
#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import sys
from pycocotools.coco import COCO
import math
import torch.utils.data as data
import numpy as np
import os
import requests
import time

from utils import train, validate, save_epoch, early_stopping
from data_loader import get_loader
from model import EncoderCNN, DecoderRNN

# Set values for the training variables
batch_size = 32         # batch size
vocab_threshold = 5     # minimum word count threshold
vocab_from_file = True  # if True, load existing vocab file
embed_size = 256        # dimensionality of image and word embeddings
hidden_size = 512       # number of features in hidden state of the RNN decoder
num_epochs = 50         # number of training epochs


# <a id='1c'></a>
# ### Image transform
# 
# When setting this transform, we keep two things in mind:
# - the images in the dataset have varying heights and widths, and 
# - since we are using a pre-trained model, we must perform the corresponding appropriate normalization.
# 
# **Training set**: As seen in the following code cell, we will set the transform for training set as follows:
# 
# ```python
# transform_train = transforms.Compose([ 
#     transforms.Resize(256),                          # smaller edge of image resized to 256
#     transforms.RandomCrop(224),                      # get 224x224 crop from random location
#     transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
#     transforms.ToTensor(),                           # convert the PIL Image to a tensor
#     transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
#                          (0.229, 0.224, 0.225))])
# ```
# 
# According to [this page](https://pytorch.org/docs/master/torchvision/models.html), like other pre-trained models, ResNet expects input images normalized as follows: 
# - The images are expected to have width and height of at least 224. The first and second transformations resize and crop the images to 224 x 224:
# ```python
# transforms.Resize(256),                          # smaller edge of image resized to 256
# transforms.RandomCrop(224),                      # get 224x224 crop from random location
# ```
# - The images have to be converted from numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]:
# ```python
# transforms.ToTensor(),                           # convert the PIL Image to a tensor
# ```
# - Then they are normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]. This is achieved using the last transformation step:
# ```python
# transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
#                          (0.229, 0.224, 0.225))
# ```
# 
# The data augmentation step `transforms.RandomHorizontalFlip()` improves the accuracy of the image classification task as mentioned in [this paper](http://cs231n.stanford.edu/reports/2017/pdfs/300.pdf).
# 
# **Validation set**: We won't use the image augmentation step, i.e. RandomHorizontalFlip(), and will use CenterCrop() instead of RandomCrop().

# In[3]:


# Define a transform to pre-process the training images
transform_train = transforms.Compose([ 
    transforms.Resize(256),                          # smaller edge of image resized to 256
    transforms.RandomCrop(224),                      # get 224x224 crop from random location
    transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
    transforms.ToTensor(),                           # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))])

# Define a transform to pre-process the validation images
transform_val = transforms.Compose([ 
    transforms.Resize(256),                          # smaller edge of image resized to 256
    transforms.CenterCrop(224),                      # get 224x224 crop from the center
    transforms.ToTensor(),                           # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))])


# <a id='1d'></a>
# ### Data loader
# We will build data loaders for training and validation sets, applying the above image transforms. We will then get the size of the vocabulary from the `train_loader`, and use it to initialize our `encoder` and `decoder`.

# In[4]:


# Build data loader, applying the transforms
train_loader = get_loader(transform=transform_train,
                         mode='train',
                         batch_size=batch_size,
                         vocab_threshold=vocab_threshold,
                         vocab_from_file=vocab_from_file)
val_loader = get_loader(transform=transform_val,
                         mode='val',
                         batch_size=batch_size,
                         vocab_threshold=vocab_threshold,
                         vocab_from_file=vocab_from_file)


# The size of the vocabulary
vocab_size = len(train_loader.dataset.vocab)

# Initialize the encoder and decoder
encoder = EncoderCNN(embed_size)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)

# Move models to GPU if CUDA is available
if torch.cuda.is_available():
    encoder.cuda()
    decoder.cuda()


# <a id='1e'></a>
# ### Loss function, learnable parameters and optimizer
# 
# **Loss function**: We will use `CrossEntropyLoss()`.
# 
# **Learnable parameters**: According to [this paper](https://arxiv.org/pdf/1411.4555.pdf), the "loss is minimized w.r.t. all the parameters of the LSTM, the top layer of the image embedder CNN and word embeddings." We will follow this strategy and choose the parameters accordingly. Since we also added a Batch Normalization layer, we will optimize its parameters too. This makes sense for two reasons:
# - the EncoderCNN in this project uses ResNet which has been pre-trained on an image classification task. So we don't have to optimize the parameters of the entire network again for a similar image classification task. We only need to optimize the top layer whose outputs are fed into the DecoderRNN.
# - the DecoderRNN is not a pre-trained network, so we have to optimize all its parameters.
# 
# **Optimizer**: According to [this paper](https://arxiv.org/pdf/1502.03044.pdf), Adam optimizer works best on the MS COCO Dataset. Therefore, we will use it.

# In[5]:


# Define the loss function
criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()

# Specify the learnable parameters of the model
params = list(decoder.parameters()) + list(encoder.embed.parameters()) + list(encoder.bn.parameters())

# Define the optimizer
optimizer = torch.optim.Adam(params=params, lr=0.001)


# <a id='step2'></a>
# ## Step 2: Train and Validate the Model
# 
# At the beginning of this notebook, we have imported the `train` fuction and the `validate` function from `utils.py`. To figure out how well our model is doing, we will print out the training loss and perplexity during training. We will try to minimize overfitting by assessing the model's performance, i.e. the Bleu-4 score, on the validation dataset. 
# 
# It will take a long time to train and validate the model. Therefore we will split the training procedure into two parts: first, we will train the model for the first time and save the it every 100 steps; then we will resume, as many times as we would like or until the early stopping criterion is satisfied. We will save the model and optimizer weights in the `models` subdirectory. We will do the same for the validation procedure.
# 
# First, let's calculate the total number of training and validation steps per epoch.

# In[6]:


# Set the total number of training and validation steps per epoch
total_train_step = math.ceil(len(train_loader.dataset.caption_lengths) / train_loader.batch_sampler.batch_size)
total_val_step = math.ceil(len(val_loader.dataset.caption_lengths) / val_loader.batch_sampler.batch_size)
print ("Number of training steps:", total_train_step)
print ("Number of validation steps:", total_val_step)


# <a id='2a'></a>
# ### Train for the first time
# 
# Run the below cell if training for the first time or training continously without break. To resume training, skip this cell and run the one below it.

# In[ ]:


# Keep track of train and validation losses and validation Bleu-4 scores by epoch
train_losses = []
val_losses = []
val_bleus = []
# Keep track of the current best validation Bleu score
best_val_bleu = float("-INF")

start_time = time.time()
for epoch in range(1, num_epochs + 1):
    train_loss = train(train_loader, encoder, decoder, criterion, optimizer, 
                       vocab_size, epoch, total_train_step)
    train_losses.append(train_loss)
    val_loss, val_bleu = validate(val_loader, encoder, decoder, criterion,
                                  train_loader.dataset.vocab, epoch, total_val_step)
    val_losses.append(val_loss)
    val_bleus.append(val_bleu)
    if val_bleu > best_val_bleu:
        print ("Validation Bleu-4 improved from {:0.4f} to {:0.4f}, saving model to best-model.pkl".
               format(best_val_bleu, val_bleu))
        best_val_bleu = val_bleu
        filename = os.path.join("./models", "best-model.pkl")
        save_epoch(filename, encoder, decoder, optimizer, train_losses, val_losses, 
                   val_bleu, val_bleus, epoch)
    else:
        print ("Validation Bleu-4 did not improve, saving model to model-{}.pkl".format(epoch))
    # Save the entire model anyway, regardless of being the best model so far or not
    filename = os.path.join("./models", "model-{}.pkl".format(epoch))
    save_epoch(filename, encoder, decoder, optimizer, train_losses, val_losses, 
               val_bleu, val_bleus, epoch)
    print ("Epoch [%d/%d] took %ds" % (epoch, num_epochs, time.time() - start_time))
    if epoch > 5:
        # Stop if the validation Bleu doesn't improve for 3 epochs
        if early_stopping(val_bleus, 3):
            break
    start_time = time.time()


# <a id='2b'></a>
# ### Resume training
# 
# Resume training if having trained and saved the model. There are two types of data loading for training depending on where we are in the process: 
# 1. We will load a model from the latest training step if we are in the middle of the process and have previously saved a model, e.g. train-model-14000.pkl which means model was saved for epoch 1 at training step 4000.
# 2. We will load a model saved by the below validation process after completing validating one epoch. This is when we start to train the next epoch. Therefore, we need to reset `start_loss` and `start_step` to 0.0 and 1 respectively.
# 
# We will modify the code cell below depending on where we are in the training process.

# In[ ]:

'''
# Load the last checkpoints
checkpoint = torch.load(os.path.join('./models', 'train-model-76500.pkl'))

# Load the pre-trained weights
encoder.load_state_dict(checkpoint['encoder'])
decoder.load_state_dict(checkpoint['decoder'])
optimizer.load_state_dict(checkpoint['optimizer'])

# Load start_loss from checkpoint if in the middle of training process; otherwise, comment it out
start_loss = checkpoint['total_loss']
# Reset start_loss to 0.0 if starting a new epoch; otherwise comment it out
#start_loss = 0.0

# Load epoch. Add 1 if we start a new epoch
epoch = checkpoint['epoch']
# Load start_step from checkpoint if in the middle of training process; otherwise, comment it out
start_step = checkpoint['train_step'] + 1
# Reset start_step to 1 if starting a new epoch; otherwise comment it out
#start_step = 1

# Train 1 epoch at a time due to very long training time
train_loss = train(train_loader, encoder, decoder, criterion, optimizer, 
                   vocab_size, epoch, total_train_step, start_step, start_loss)

'''
# Now that we have completed training an entire epoch, we will save the necessary information. We will load pre-trained weights from the last train step `train-model-{epoch}12900.pkl`, `best_val_bleu` from `best-model.pkl` and the rest from `model-{epoch}.pkl`). We will append `train_loss` to the list `train_losses`. Then we will save the information needed for the epoch.

# In[8]:

'''
# Load checkpoints
train_checkpoint = torch.load(os.path.join('./models', 'train-model-712900.pkl'))
epoch_checkpoint = torch.load(os.path.join('./models', 'model-6.pkl'))
best_checkpoint = torch.load(os.path.join('./models', 'best-model.pkl'))

# Load the pre-trained weights and epoch from the last train step
encoder.load_state_dict(train_checkpoint['encoder'])
decoder.load_state_dict(train_checkpoint['decoder'])
optimizer.load_state_dict(train_checkpoint['optimizer'])
epoch = train_checkpoint['epoch']

# Load from the previous epoch
train_losses = epoch_checkpoint['train_losses']
val_losses = epoch_checkpoint['val_losses']
val_bleus = epoch_checkpoint['val_bleus']

# Load from the best model
best_val_bleu = best_checkpoint['val_bleu']

train_losses.append(train_loss)
print (train_losses, val_losses, val_bleus, best_val_bleu)
print ("Training completed for epoch {}, saving model to train-model-{}.pkl".format(epoch, epoch))
filename = os.path.join("./models", "train-model-{}.pkl".format(epoch))
save_epoch(filename, encoder, decoder, optimizer, train_losses, val_losses, 
           best_val_bleu, val_bleus, epoch)


# <a id='2c'></a>
# ### Validation
# 
# We will do validation for an epoch once we have trained and saved the model for that epoch. There are two types of data loading for validation depending on where we are in the process: 
# 1. We will load a model from the latest validation step if we are in the middle of the process and have previously saved a model, e.g. val-model-14000.pkl which means the model was saved for epoch 1 at val step 4000.
# 2. We will load a model saved by the above training process after completing training one epoch. This is when we just start to do validation, i.e. at validation step \#1. Therefore, we need to reset `start_loss`, `start_bleu` and `start_step` to 0.0, 0.0 and 1 respectively.
# 
# We will modify the code cell below depending on where we are in the validation process.

# In[7]:


# Load the last checkpoint
checkpoint = torch.load(os.path.join('./models', 'val-model-75500.pkl'))

# Load the pre-trained weights
encoder.load_state_dict(checkpoint['encoder'])
decoder.load_state_dict(checkpoint['decoder'])

# Load these from checkpoint if in the middle of validation process; otherwise, comment them out
start_loss = checkpoint['total_loss']
start_bleu = checkpoint['total_bleu_4']
# Reset these to 0.0 if starting validation for an epoch; otherwise comment them out
#start_loss = 0.0
#start_bleu = 0.0

# Load epoch
epoch = checkpoint['epoch']
# Load start_step from checkpoint if in the middle of training process; otherwise, comment it out
start_step = checkpoint['val_step'] + 1
# Reset start_step to 1 if starting a new epoch; otherwise comment it out
#start_step = 1

# Validate 1 epoch at a time due to very long validation time
val_loss, val_bleu = validate(val_loader, encoder, decoder, criterion, 
                              train_loader.dataset.vocab, epoch, total_val_step, 
                              start_step, start_loss, start_bleu)


# Now that we have completed training and validation for an entire epoch, we will save all the necessary information. We will load most information from `train-model-{epoch}.pkl` and `best_val_bleu` from `best-model.pkl`. We will then do the following updates:
# - appending `val_bleu` and `val_loss` to the lists `val_bleus` and `val_losses` respectively
# - updating `best_val_bleu` if it is not as good as `val_bleu` we just got in the above cell
# 
# Then we will save the information needed for the epoch.

# In[8]:


# Load checkpoints
checkpoint = torch.load(os.path.join('./models', 'train-model-7.pkl'))
best_checkpoint = torch.load(os.path.join('./models', 'best-model.pkl'))

# Load the pre-trained weights
encoder.load_state_dict(checkpoint['encoder'])
decoder.load_state_dict(checkpoint['decoder'])
optimizer.load_state_dict(checkpoint['optimizer'])

# Load train and validation losses and validation Bleu-4 scores 
train_losses = checkpoint['train_losses']
val_losses = checkpoint['val_losses']
val_bleus = checkpoint['val_bleus']
best_val_bleu = best_checkpoint['val_bleu']

# Load epoch
epoch = checkpoint['epoch']    

val_losses.append(val_loss)
val_bleus.append(val_bleu)
print (train_losses, val_losses, val_bleus, best_val_bleu)

if val_bleu > best_val_bleu:
    print ("Validation Bleu-4 improved from {:0.4f} to {:0.4f}, saving model to best-model.pkl".
           format(best_val_bleu, val_bleu))
    best_val_bleu = val_bleu
    print (best_val_bleu)
    filename = os.path.join("./models", "best-model.pkl")
    save_epoch(filename, encoder, decoder, optimizer, train_losses, val_losses, 
               val_bleu, val_bleus, epoch)
else:
    print ("Validation Bleu-4 did not improve, saving model to model-{}.pkl".format(epoch))
# Save the entire model anyway, regardless of being the best model so far or not
filename = os.path.join("./models", "model-{}.pkl".format(epoch))
save_epoch(filename, encoder, decoder, optimizer, train_losses, val_losses, 
           val_bleu, val_bleus, epoch)
if epoch > 5:
    # Stop if the validation Bleu doesn't improve for 3 epochs
    if early_stopping(val_bleus, 3):
        print ("Val Bleu-4 doesn't improve anymore. Early stopping")


# <a id='2d'></a>
# ### Notes regarding model validation
# 
# - Another way to validate a model involves creating a json file such as [this one](https://github.com/cocodataset/cocoapi/blob/master/results/captions_val2014_fakecap_results.json) containing the model's predicted captions for the validation images. Then, write up a script or use one [available online](https://github.com/tylin/coco-caption) to calculate the BLEU score of the model. 
# - Other evaluation metrics (such as TEOR and Cider) are mentioned in section 4.1 of [this paper](https://arxiv.org/pdf/1411.4555.pdf). 
# 
# 
# # Next steps
# 
# A few things that we may try in the future to improve model performance:
# 
# - Adjust learning rate: make it decay over time, as in [this example](https://github.com/pytorch/examples/blob/master/imagenet/main.py).
# - Run the code on a GPU to so that we can train the model more. Tried AWS p2.xlarge; however, the datasets exceeded the storage limit.
'''
