import matplotlib. pyplot as plt
import os
import numpy as np
from transformers import *
import config
import bert


def plot_label_distribution(y):
    """Analyze label distribution of dataset"""

    if config.class_task == "reldetect":

        label_names = ["Visited", "Founder", "Nationality", "Wife", "PoliticalAffiliation", "JobTitle", "Education",
                       "Employer", "Awarded", "BirthPlace", "DeathPlace"]
        all_relations = np.sum(y, 0)
        plt.bar(range(len(all_relations)), all_relations, alpha=0.5)
        plt.xticks(ticks=np.arange(len(all_relations)), labels=label_names, fontsize=10)
        plt.savefig('label-distribution-' + config.class_task + '.png')
        plt.clf()

        # plot number of relation types per sentence
        rels_per_sentence = [sum(s) for s in y]
        plt.hist(rels_per_sentence, bins=max(rels_per_sentence), alpha=0.5)
        plt.xticks(fontsize=10)
        plt.xlabel('no. of relations')
        plt.ylabel('no. of sentences')
        plt.savefig('relation-distribution-' + config.class_task + '.png')

    else:
        plt.hist(y, bins=len(set(y)), alpha=0.5)
        plt.xticks(rotation=90, fontsize=7)
        plt.savefig('label-distribution-' + config.class_task + '.png')


def load_glove_embeddings(vocab_size, word_index, EMBEDDING_DIM):

    print('Indexing word vectors.')

    embeddings_index = {}
    with open(os.path.join(
            config.base_dir + 'eego/feature_extraction/embeddings/glove.6B.'+str(EMBEDDING_DIM)+'d.txt')) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            embeddings_index[word] = coefs

    print('Found %s word vectors.' % len(embeddings_index))

    # prepare embedding matrix
    num_words = min(vocab_size, len(word_index) + 1)

    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= vocab_size:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

"""
def load_bert_embeddings(X, max_length, bert_dim):
    # Allocate a pipeline for feature extraction (= generates a tensor representation for the input sequence)
    # https://github.com/huggingface/transformers#quick-tour-of-pipelines

    X_bert_states_padded = []

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = TFBertModel.from_pretrained('bert-base-uncased')

    for sent in X:

        input_ids = tf.constant(tokenizer.encode(sent, max_length=max_length, add_special_tokens=True, pad_to_max_length=True))[None, :]  # Batch size 1
        #print(input_ids)

        outputs = model(input_ids)

        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
        cls_embeddings = last_hidden_states[0] # Use hidden states of the [CLS] token of the last layer as sentence embedding for classification
        #print(last_hidden_states)
        #print(cls_embeddings.shape)
        X_bert_states_padded.append(cls_embeddings)

    #print(len(X_bert_states_padded))
    return np.asarray(X_bert_states_padded)
"""



def createTokenizer():
    """initialize Bert tokenizer"""
    # bert implementationadapted from here:
    # https://medium.com/@brn.pistone/bert-fine-tuning-for-tensorflow-2-0-with-keras-api-9913fc1348f6

    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #return tokenizer

    modelsFolder = os.path.join(config.modelBertDir, "uncased_L-12_H-768_A-12")
    vocab_file = os.path.join(modelsFolder, "vocab.txt")

    tokenizer = bert.bert_tokenization.FullTokenizer(vocab_file, do_lower_case=True)
    return tokenizer


def prepare_sequences_for_bert(X):
    """ tokenize sentences and add special tokens needed for Bert"""


    tokenizer = createTokenizer()

    tokens = map(tokenizer.tokenize, X)
    tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], tokens)
    token_ids = list(map(tokenizer.convert_tokens_to_ids, tokens))

    token_ids = np.array(list(token_ids))

    return token_ids


def createBertLayer():
    global bert_layer

    # todo: model not the same as for tokenizer -- does it matter?
    #bertDir = os.path.join(config.modelBertDir, "multi_cased_L-12_H-768_A-12")
    bertDir = os.path.join(config.modelBertDir, "uncased_L-12_H-768_A-12")

    bert_params = bert.params_from_pretrained_ckpt(bertDir)

    bert_layer = bert.BertModelLayer.from_params(bert_params, name="bert_layer")

    bert_layer.apply_adapter_freeze()

    print("Bert layer created")

    return bert_layer
