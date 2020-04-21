import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
import numpy as np
import config
from transformers import BertTokenizer
from transformers import TFBertModel
from sklearn.preprocessing import MinMaxScaler



def plot_label_distribution(y):
    """Analyze label distribution of dataset"""

    if config.class_task == "reldetect":

        label_names = ["Visited", "Founder", "Nationality", "Wife", "PoliticalAffiliation", "JobTitle", "Education",
                       "Employer", "Awarded", "BirthPlace", "DeathPlace"]

        all_relations = np.sum(y, 0)
        #print(all_relations)
        #print(label_names)
        plt.clf()
        # todo: make plots a bit nicer :)
        plt.bar(range(len(all_relations)), all_relations, alpha=0.5)
        plt.xticks(rotation=45, ticks=np.arange(len(all_relations)), labels=label_names, fontsize=7)
        plt.savefig('label-distribution-' + config.class_task + '.png')
        plt.clf()

        # plot number of relation types per sentence
        rels_per_sentence = {}
        for s in y:
            if sum(s) not in rels_per_sentence:
                rels_per_sentence[sum(s)] = 1
            else:
                rels_per_sentence[sum(s)] += 1
        plt.bar(range(len(rels_per_sentence)), rels_per_sentence.values(), alpha=0.5)
        plt.xticks(fontsize=10, ticks=np.arange(len(rels_per_sentence)), labels=list(range(len(rels_per_sentence))))
        plt.xlabel('no. of relations')
        plt.ylabel('no. of sentences')
        plt.tight_layout()
        plt.savefig('relation-distribution-' + config.class_task + '.png')
        plt.clf()

    else:
        plt.hist(y, bins=len(set(y)), alpha=0.5)
        plt.xticks(rotation=90, fontsize=7)
        plt.savefig('label-distribution-' + config.class_task + '.png')
        plt.tight_layout()
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

    print("glove matrix:", embedding_matrix.shape)
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
    # Tokenize all of the sentences and map the tokens to their word IDs.
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


def scale_feature_values(X):
    """Scale eye-tracking and EEG feature values"""

    scaler = MinMaxScaler(feature_range=(0, 1))
    feat_values = []

    for feat in range(len(X[0][0])):
        for sentence in X:
            for token in sentence:
                feat_values.append(token[feat])

    # train the normalization
    feat_values = np.array(feat_values).reshape(-1, 1)
    scaler = scaler.fit(feat_values)
    print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))
    # normalize the dataset and print
    normalized = scaler.transform(feat_values)

    # add normalized values back to feature list
    i = 0
    for sentence in X:
        for token in sentence:
            token[feat] = normalized[i]
            i += 1

    return X


def plot_confusion_matrix(cm):
    import matplotlib.pyplot as plt
    from mlxtend.plotting import plot_confusion_matrix

    fig, ax = plot_confusion_matrix(conf_mat=cm, colorbar=True,
                                    show_absolute=True,
                                    show_normed=True)
    #ax.set_xticklabels([''] + target_names)
    #ax.set_yticklabels([''] + target_names)
    plt.title("Confusion matrix: " + config.class_task + ", " + config.feature_set)
    plt.savefig("CM_test.pdf")
    # plt.show()