from . import data_loading_helpers as dh
import config
from spacy.lang.en import English
nlp = English()
# Create a Tokenizer with the default settings for English
# including punctuation rules and exceptions
tokenizer = nlp.Defaults.create_tokenizer(nlp)


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
            spacy_tokens = tokenizer(sent)

            # for sentiment
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

