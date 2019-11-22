import os
import torch
import bert_tokenization
import pickle
from collections import Counter
from lex_parser import Lex_parser

class Dictionary(object):
    """ 
    word2idx = {word0: 0, word1: 1, ....}
    idx2word = [word0, word1, ....]
    counter = {0: 10, 1: 11}
    """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):

    def __init__(self, path):
        """
        There are three files: train.txt, valid.txt, and test.txt in that dictionary.

        :param path: e.g., data_process.py
        """
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1
        # print(ids[0:10])
        # print("The shape of {}".format(path), ids.shape)
        return ids


class BertCorpus(object):
    def __init__(self, path, bertTokenizer):
        """
        There are three files: train.txt, valid.txt, and test.txt in that dictionary.

        :param path: e.g., data_process.py
        """
        self.dictionary = Dictionary()
        self.bert_tokenizer = bertTokenizer

        self.train, self.train_tags = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid, self.valid_tags = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test, self.test_tags = self.tokenize(os.path.join(path, 'test.txt'))
        # self.test_tags = self.convert_syntax_tags(os.path.join(path, 'test.txt'), self.test)


    def tokenize(self, path, store=True):
        """
        :param path:
        :param store:
        :return:
        """
        bert_pickle_file = path.replace('.txt',"_bert.pickle")
        lex_pickle_file = path.replace('.txt', "_lex.pickle")
        bert_res, lex_res = None, None
        if os.path.exists(bert_pickle_file):
            with open(bert_pickle_file, 'rb') as f:
                bert_res = pickle.load(f)
                print(bert_pickle_file, "is already there.")

        if os.path.exists(lex_pickle_file):
            with open(lex_pickle_file, 'rb') as f:
                lex_res = pickle.load(f)
                print(lex_pickle_file, "is already there.")

        if bert_res is not None and lex_res is not None:
            return bert_res, lex_res

        assert os.path.exists(path)
        # Add words to the dictionary
        if not self.bert_tokenizer:
            raise Exception("bert tokenizer is not specified")

        parser = Lex_parser()

        token_ids = []
        token_tag_ids = []
        with open(path, 'r') as f:
            for line in f:
                # sentence = " ".join([line.strip('\n'), '<eos>']) # should we add '<eos>' at the end of each sentence, currently we add <eos> and after bert tokenization, it will be [UNK]
                # tokenized_text = self.bert_tokenizer.tokenize(sentence, process_N=True, seperate_punc=True, keep_eos=True)  # numbers in the text are 'N's, after bert tokenization, they will be '[UNK]'
                # indexed_tokens = self.bert_tokenizer.convert_tokens_to_ids(tokenized_text)

                # currently use modified pretrained bert
                sentence = " ".join([line.strip('\n'), '<eos>']) # should we add '<eos>' at the end of each sentence, currently we add <eos> and after bert tokenization, it will be [UNK]
                tokenized_text = self.bert_tokenizer.tokenize(sentence)  # numbers in the text are 'N's, after bert tokenization, they will be '[UNK]'

                curr_token_ids = self.bert_tokenizer.convert_tokens_to_ids(tokenized_text)
                parsed_words = dict(parser.convert_sentence_to_tags(sentence))
                # print("parsed words", parsed_words)
                curr_token_tag_ids = self.bert_tokenizer.convert_token_ids_to_tag_ids(curr_token_ids, parsed_words, parser.tag_to_id)
                assert len(curr_token_ids) == len(curr_token_tag_ids)

                token_ids += curr_token_ids
                token_tag_ids += curr_token_tag_ids

        token_ids = torch.tensor(token_ids)
        token_tag_ids = torch.tensor(token_tag_ids)

        if store:
            with open(bert_pickle_file, 'wb') as f:
                pickle.dump(token_ids, f)
            with open(lex_pickle_file, 'wb') as f:
                pickle.dump(token_tag_ids, f)
        return token_ids, token_tag_ids

    # # convert bert
    # def convert_syntax_tags(self, path, bert_tokens, store=True):
    #     pickle_file = path.replace('.txt', "_lex.pickle")
    #     if os.path.exists(pickle_file):
    #         with open(pickle_file, 'rb') as f:
    #             res = pickle.load(f)
    #             print(pickle_file, "is already there.", type(res))
    #             return res
    #
    #     tag_ids = []
    #     parser = Lex_parser()
    #
    #     with open(path, 'r') as f:
    #         for line in f:
    #             sentence = " ".join([line.strip('\n'), '<eos>'])
    #             token_tags = parser.convert_sentence_to_tags(sentence)
    #             # print("indexed_tokens", len(ids))
    #     self.bertTokenizer.convert_token_ids_to_tag_ids()
    #     ids.extend(indexed_tokens)
    #     res = torch.tensor(ids)
    #
    #     if store:
    #         with open(pickle_file, 'wb') as f:
    #             pickle.dump(tag_ids, f)
    #     return tag_ids

# class SyntaxCorpus(object):
#     def __init__(self, path, bertTokenizer):
#         """
#         There are three files: train.txt, valid.txt, and test.txt in that dictionary.
#
#         :param path: e.g., data_process.py
#         """
#         self.parser= Lex_parser()
#         self.bertTokenizer = bertTokenizer
#         self.train = self.tokenize(os.path.join(path, 'train.txt'))
#         self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
#         self.test = self.tokenize(os.path.join(path, 'test.txt'))
#
#     def tokenize(self, path, store=True):
#         """
#         :param path: the path of the file
#         :param vocab: the name of pretrained bert vovabs we use, by default we will use "bert-base-uncased"
#         :return:
#         """
#         pickle_file = path.replace('.txt', "_lex.pickle")
#         if os.path.exists(pickle_file):
#             with open(pickle_file, 'rb') as f:
#                 res = pickle.load(f)
#                 print(pickle_file, "is already there.", type(res))
#                 return res
#
#         assert os.path.exists(path)
#         # Add words to the dictionary
#         ids = []
#         with open(path, 'r') as f:
#             for line in f:
#                 sentence = " ".join([line.strip('\n'), '<eos>'])
#                 indexed_tokens = self.parser.convert_sentence_to_tags(sentence)
#                 self.bertTokenizer.convert_token_ids_to_tag_ids()
#                 ids.extend(indexed_tokens)
#                 # print("indexed_tokens", len(ids))
#         res = torch.tensor(ids)
#
#         if store:
#             with open(pickle_file, 'wb') as f:
#                 pickle.dump(res, f)
#         return res


def batchify(data, bsz, cuda):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if cuda:
        data = data.cuda()
    # print("data size", type(data), data.shape ) #, data)
    return data


if __name__ == '__main__':
    # This corpus is used in
    corpus = Corpus('data/penn')
    print("penn corpus in original paper", corpus.train.shape, corpus.valid.shape, corpus.test.shape)

    bert_tokenizer = bert_tokenization.BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    bertCorpus = BertCorpus('data/penn', bert_tokenizer)
    print("bert corpus", bertCorpus.train.shape, bertCorpus.valid.shape, bertCorpus.test.shape)

    print("lex corpus", bertCorpus.train_tags.shape, bertCorpus.valid_tags.shape, bertCorpus.test_tags.shape)

    print("bert corpus[20:40]", bertCorpus.train[20:40])
    print("lex corpus[20:40]", bertCorpus.train_tags[20:40])

    # batchify(bertCorpus.train, bsz=20, cuda=False)
    # batchify(bertCorpus.test, bsz=20, cuda=False)
    # batchify(bertCorpus.valid, bsz=20, cuda=False)

    # syntaxCorpus = SyntaxCorpus('data/penn')
    # print("syntax tag corpus", syntaxCorpus.train.shape, syntaxCorpus.valid.shape, syntaxCorpus.test.shape)