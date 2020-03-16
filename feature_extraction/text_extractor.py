from . import data_loading_helpers as dh
import config
import nltk


def extract_sentences(sentence_data, sentence_dict):
    """extract tokens of all sentences."""

    for tup in sentence_data:

        f = tup[0]
        s_data = tup[1]

        rawData = s_data['rawData']
        contentData = s_data['content']

        for idx in range(len(rawData)):
            obj_reference_content = contentData[idx][0]
            sent = dh.load_matlab_string(f[obj_reference_content])
            split_tokens = sent.split()
            # todo: add tokenized version
            spacy_tokens = nltk.word_tokenize(sent)
            #print(spacy_tokens)
            if "excelling" in spacy_tokens:
                print(spacy_tokens)

            # for sentiment
            if config.class_task.startswith('sentiment'):
                if sent not in sentence_dict:
                    sentence_dict[sent] = split_tokens
                else:
                    print('duplicate!')

            # for ner
            if config.class_task == "ner":
                if sent not in sentence_dict:
                    sentence_dict[sent] = spacy_tokens

                else:
                    print('duplicate!')

