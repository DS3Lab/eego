from . import data_loading_helpers as dh
import config
import nltk


def extract_sentences(sentence_data, sentence_dict):
    """extract tokens of all sentences."""

    for tup in sentence_data:

        # read Matlab v7.3 structures
        f = tup[0]
        s_data = tup[1]
        rawData = s_data['rawData']
        contentData = s_data['content']

        for idx in range(len(rawData)):
            obj_reference_content = contentData[idx][0]
            sent = dh.load_matlab_string(f[obj_reference_content])
            # whitespace tokenization
            split_tokens = sent.split()
            # linguistic tokenization
            # spacy_tokens = nltk.word_tokenize(sent)

            # for sentiment and relation detection
            if config.class_task.startswith('sentiment') or config.class_task == "reldetect":
                if sent not in sentence_dict:
                    sentence_dict[sent] = split_tokens
        print(len(sentence_dict))
