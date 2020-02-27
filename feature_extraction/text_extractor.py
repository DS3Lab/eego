from . import data_loading_helpers as dh


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
        tokens = sent.split()
        # todo: add tokenized version

        if sent not in sentence_dict:
            sentence_dict[sent] = tokens
        #else:
         #   print('duplicate!')