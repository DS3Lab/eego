import matplotlib. pyplot as plt
import os
import numpy as np
import config
import bert
from transformers import BertTokenizer
from transformers import TFBertModel


def plot_label_distribution(y):
    """Analyze label distribution of dataset"""

    if config.class_task == "reldetect":

        label_names = ["Visited", "Founder", "Nationality", "Wife", "PoliticalAffiliation", "JobTitle", "Education",
                       "Employer", "Awarded", "BirthPlace", "DeathPlace"]
        all_relations = np.sum(y, 0)
        plt.clf()
        plt.bar(range(len(all_relations)), all_relations, alpha=0.5)
        plt.xticks(ticks=np.arange(len(all_relations)), labels=label_names, fontsize=10)
        plt.savefig('label-distribution-' + config.class_task + '.png')
        plt.clf()

        # plot number of relation types per sentence
        rels_per_sentence = [sum(s) for s in y]
        plt.hist(rels_per_sentence, bins=max(rels_per_sentence), alpha=0.5)
        plt.xticks(rotation=90, fontsize=10)
        plt.xlabel('no. of relations')
        plt.ylabel('no. of sentences')
        plt.savefig('relation-distribution-' + config.class_task + '.png')
        plt.clf()

    else:
        plt.hist(y, bins=len(set(y)), alpha=0.5)
        plt.xticks(rotation=90, fontsize=7)
        plt.savefig('label-distribution-' + config.class_task + '.png')
        plt.clf()


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


def get_bert_max_len(X):

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    max_len = 0
    # For every sentence...
    for sent in X:
        # Tokenize the text and add [CLS] and [SEP] tokens.
        input_ids = tokenizer.encode(sent, add_special_tokens=True)
        # Update the maximum sentence length.
        max_len = max(max_len, len(input_ids))
    print('Max Bert sentence length: ', max_len)

    return max_len


def prepare_sequences_for_bert_with_mask(X, max_length):
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in X:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = max_length,           # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                       )
        
        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])
        
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = np.vstack(input_ids)

    attention_masks = np.vstack(attention_masks)

    return input_ids, attention_masks


def create_new_bert_layer():
    bert = TFBertModel.from_pretrained("bert-base-uncased")
    return bert
