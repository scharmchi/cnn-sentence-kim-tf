####################
# Original taken from https://github.com/dennybritz/cnn-text-classification-tf/blob/master/data_helpers.py
####################
"""
Originally taken from http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/

The preprocessing does the following:
    1. Load positive and negative sentences from the raw data files.
    2. Clean the text data using the same code as the original paper.
    3. Pad each sentence to the maximum sentence length, which turns out to be 59.
        We append special <PAD> tokens to all other sentences to make them 59 words.
        Padding sentences to the same length is useful because it allows us to efficiently batch our data
        since each example in a batch must be of the same length.
    4. Build a vocabulary index and map each word to an integer between 0 and 18,765
        (the vocabulary size). Each sentence becomes a vector of integers.
"""
import numpy as np
import re
import itertools
import sys
from collections import Counter


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_MR_data_and_labels():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open("./SST-2/train/train_pos_sst", "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open("./SST-2/train/train_neg_sst", "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    x_text = [s.split(" ") for s in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def load_TREC_data_and_labels(is_train):
    """
    Loads TREC data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    if is_train:
        # Load data from files
        examples = list(open("./TREC/shitty_train_test", "r").readlines())
        examples = [s.strip() for s in examples]
    else:
        examples = list(open("./TREC/TREC_10.label", "r").readlines())
        examples = [s.strip() for s in examples]
        # examples = [long for long in examples if len(long.split(" ")) > 5]
    labels = []
    for s in examples:
        labels.append(s[:s.find(":")])
    examples_wo_labels = [s[s.find(" ") + 1:].strip() for s in examples]
    # print(train_examples_wo_labels)
    # Split by words
    x_text = [clean_str(sent) for sent in examples_wo_labels]
    x_text = [s.split(" ") for s in x_text]
    # Generate labels
    y = []
    for label in labels:
        if label == "DESC":
            y.append([1, 0, 0, 0, 0, 0])
        elif label == "ENTY":
            y.append([0, 1, 0, 0, 0, 0])
        elif label == "ABBR":
            y.append([0, 0, 1, 0, 0, 0])
        elif label == "HUM":
            y.append([0, 0, 0, 1, 0, 0])
        elif label == "NUM":
            y.append([0, 0, 0, 0, 1, 0])
        elif label == "LOC":
            y.append([0, 0, 0, 0, 0, 1])
    return [x_text, y]


def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def load_data(is_train=True):
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_TREC_data_and_labels(is_train)
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    # print("num_batches_per_epoch={}".format(num_batches_per_epoch))
    for epoch in range(num_epochs):
        # print("\nEpoch number: " + str(epoch + 1) + "\n")
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
