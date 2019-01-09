# Argumentation Mining

This is a project from Lecture Deep Learning for NLP of University Darmstadt at June 11, 2018. It posted as a compitation at https://competitions.codalab.org/competitions/19092#learn_the_details.

Argumentation Mining is the problem of mining arguments in a text. According to some argumentation scholars, an argument consists of claims and premises. While the definition of claims and premises is to some degree vague, claims can be thought of as statements that require justification such as:
 **Lionel Messi is the best player in the history of football.**
A premise may give such a justification:
 **This is because he is the most complete soccer player, with the best technical skills.**

This project aim to mine components of an argument from text. These components are claims, premises, and major claims (a major claim can be thought of as a summary of many different claims). The training data that we use labels each token in a text with a BIO-tag (begin, inside, outside) as well as the type of the argument component: claim, major-claim, premise.
For example, the above two sentences might be labeled as (after tokenization):![image-20190105220005817](https://github.com/ThomasSYT/NLP-share-task/raw/master/img/image-20190105220005817.png)

## environment and tools

Python 3.6

Jupyter

Numpy

Keras

Pandas

Matplotlib

Sklearn

Before you run the code, please download the glove.6B.50d.txt from https://nlp.stanford.edu/projects/glove/ and add it into embedding folder.

## Data Formats

The data file contains several documents, i.e. one or more sentences. Document boundaries are represented by empty lines. A document itself consists of two tab-separated columns, where every row contains a token and its label for the task. Tokens are postpended with a language flag (such as _en). Corresponding to the task and the BIO tagging scheme, the possible values for the labels are
**O, B-Claim, I-Claim, B-MajorClaim, I-MajorClaim, B-Premise, I-Premise** 

Files end with a single empty line.

#### Preprocess of Data File

The process of data file includes 3 step:
1. Split the file into documents(Each documents is a list of tokens).
2. Remove the language identifier ‘en’.
3. Split the tokens and labels.

## Baseline Model

We choose a multi-layer perceptron to deal with sequential data as baseline model.

## Bi-LSTM Model

### Input

The input to the model is a sequence of indexes of tokens and one- hot encoding of their BIO-tags (begin, inside, outside) with the type of the argument component: claim, major-claim, premise.  Because mini-batch is used, in order to prevent the last batch from being vacant, we add the “PADDING” logo as a supplement. 

### Embedding Layer

Here, we choose [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/) as the Embedding model.  The researchers behind GloVe method provide a suite of pre-trained word embeddings on their website released under a public domain license. The smallest package(glove.6B.zip) was trained on a dataset of one billion tokens (words) with a vocabulary of 400 thousand words. There are some different embedding vector sizes, we choose 300 dimensions because it can reach a better training result. 

In addition, we add PADDING with 300 zeros and OOV (out of vocabulary) with 300 random number between 1 and -1 to the end of the embedding dictionary.  After this, the shape of embedding_matrix is 400002 * 300. 

Because we set mask_zero to true, which means that the '0' in the input is treated as a 'padding' value that should be ignored, so we further modified embedding-matrix to change the shape to 400003 * 300. The value of matrix index 0 is all set to 0.

### Dropout

In order to avoid over-fitting of the model, we added a dropout of 0.3 before and after the Bi-LSTM layer.

### Bi-LSTM

Through experiments, we found that the use of two layers of Bi-LSTM not only speeds up the convergence of the training model, but also produces better results.

One layer of Bi-LSTM(5 repeats):

![image-20190108175651132](https://github.com/ThomasSYT/NLP-share-task/raw/master/img/image-20190108175651132.png)

Two  layers of Bi-LSTM(the best result of 5 repeats):

![image-20190108175744469](https://github.com/ThomasSYT/NLP-share-task/raw/master/img/img-24.png)

### Dense

Since the task is sequence tagging, our Bi-LSTM should output a sequence. After the Bi-LSTM, we add a dense layer with n units to every sequence output, where n is the number of classes in the sequence tagging task plus 1. The activation function is softmax. Using the TimeDistributed wrapper enables the model to achieve many-to-many capabilities.

Here is the model hierarchy:

![seq2seq_model](https://github.com/ThomasSYT/NLP-share-task/raw/master/img/model.png)

### Hyperparameter optimization

Each experimental scene needs to be reproduced about 5 times, because even if the hyperparameters required for model training are given, the random initialization of the LSTM model may cause a huge difference between the model training results.

By examining the variation of model performance with the number of iterations (epochs) under different model hyperparameters, we can obtain some hyperparameter adjustment intervals or directions that may improve the performance of the model.

After the training of each batch of data sets, the scores on the test set will be printed out. For example, we show the F1score curve through a line chart after the end of the run.

Printout performance metrics at the end of each batch can help us better understand the current state of the model.

### Prediction

We use Adam as optimizer and categorical crossentropy as loss function. Because the datasets have an imbalanced label distribution, we use macro-F1 to evaluate the result.

Meanwhile, we observe some rules of the training data set:

1. If a sentence ends not with a punctuation like ‘.’ and ‘?’, then the all tokens of this sentence are with a BIO-tag ‘O’.
2. All punctuation ‘.’ is with a BIO-tag ‘O’.

So in the prediction phase, we also change the labels for above situations.

### Result

For the 2 baseline model, we only use the 50-dimensional embeddings (glove.6B.50d.txt). In order to improve the performance we also use the 300-dimensional embeddings for the main model Bi-LSTM.

Configurations for each model:

1. Baseline: window size: 30; batch size: 1000; epochs:30; hidden units: 50.
2. Bi-LSTM: batch size:16; epoch:70; hidden units: 280; dropout: 0.3.

The best f1 scores of each model are shown as followed:

| **Model**     | **f1 on dev.(en)** |
| ------------- | ------------------ |
| baseline(50d) | 0.3884             |
| Bi-LSTM(50d)  | 0.6149             |
| Bi-LSTM(300d) | 0.7195             |

In the trial phase, the results of the current model rank first in the team displayed on the site(https://competitions.codalab.org/competitions/19092#results).

![image-20190108180414971](https://github.com/ThomasSYT/NLP-share-task/raw/master/img/image-20190108180414971.png)