import os
import torch
import bert_tokenization
from collections import Counter

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
        # eos_id = self.dictionary.word2idx['<eos>']
        # print("idx of eos", eos_id)
        # print("counter of eos", self.dictionary.counter[eos_id])

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
    def __init__(self, path):
        """
        There are three files: train.txt, valid.txt, and test.txt in that dictionary.

        :param path: e.g., data_process.py
        """
        self.dictionary = Dictionary()
        self.bert_tokenizer = None
        self.never_split = ("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))


    def tokenize(self, path, vocab="bert-base-uncased"):
        """
        :param path: the path of the file
        :param vocab: the name of pretrained bert vovabs we use, by default we will use "bert-base-uncased"
        :return:
        """
        assert os.path.exists(path)
        # Add words to the dictionary
        if not self.bert_tokenizer:
            self.bert_tokenizer = bert_tokenization.BertTokenizer.from_pretrained(vocab)
        ids = []
        with open(path, 'r') as f:
            for line in f:
                sentence = line[:-1] + '<eos>'  # should we add '<eos>' at the end of each sentence, currently we add <eos> and after bert tokenization, it will be [UNK]
                tokenized_text = self.bert_tokenizer.tokenize(sentence, process_N=True, seperate_punc=True, keep_eos=True)  # numbers in the text are 'N's, after bert tokenization, they will be '[UNK]'
                indexed_tokens = self.bert_tokenizer.convert_tokens_to_ids(tokenized_text)
                ids += indexed_tokens
        return torch.tensor(ids)

    def run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        if text in self.never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if bert_tokenization._is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

def batchify(data, bsz, cuda):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if cuda:
        data = data.cuda()
        print("data.cuda", type(data), data.shape ) #, data)
    return data


if __name__ == '__main__':
    bertCorpus = BertCorpus('data/penn')
    print(bertCorpus.train.shape, bertCorpus.valid.shape, bertCorpus.test.shape)
    batchify(bertCorpus.train, bsz=20, cuda=True)
    batchify(bertCorpus.test, bsz=20, cuda=True)
    batchify(bertCorpus.valid, bsz=20, cuda=True)
