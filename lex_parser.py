from nltk.parse.stanford import StanfordParser
# parser = StanfordParser()

from nltk.parse.corenlp import CoreNLPParser
from bert_tokenization import BasicTokenizer

# If you want to parse the id

# open terminal and
# cd Enbert/
# wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip
# unzip stanford-corenlp-full-2018-10-05.zip
# cd stanford-corenlp-full-2018-10-05
# java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos,lemma,ner,parse,depparse -status_port 9000 -port 9000 -timeout 15000 &

# Then run this scipt
from typing import Union

class Lex_parser:
    def __init__(self, tag_id_initialized=False, tag_id=None, uncased=True):
        self.uncased=uncased
        self.tag_id_initialized = tag_id_initialized
        if tag_id_initialized:
            self.tag_to_id = tag_id
        else:
            self.tag_to_id = {"CLSSEP": 0, "UNKNOWN": 1}
        self.parser = CoreNLPParser(url='http://localhost:9000', tagtype='pos')
        self.basic_tokenizer = BasicTokenizer()

    def tokenize(self, sentence):
        return list(self.parser.tokenize(sentence))

    def convert_sentence_to_tags(self, sentence: Union[str, list]):
        if type(sentence) == str:
            if self.uncased:
                sentence = sentence.lower()

        else:
            sentence = " ".join(sentence)
            if self.uncased:
                sentence = sentence.lower()

        sentence = self.basic_tokenizer.tokenize(sentence)


        # print("sentence here,", sentence)
        sentence = list(map(lambda x: x.upper() if x == 'i' else x, sentence))
        tags = self.parser.tag(sentence)
        # print("sentence here,", sentence)
        # print("tags here", tags)
        # exit(-2)
        if not self.tag_id_initialized:
            for tag in tags:
                if tag[1] not in self.tag_to_id:
                    self.tag_to_id[tag[1]] = len(self.tag_to_id)
        return tags

    def convert_tags_to_ids(self, tags):
        res = list(map(lambda x: self.tag_to_id[x[1]], tags))
        # print("to ids ==")
        # print(len(tags), tags)
        # print(len(res), res)
        return res


    def convert_sentence_to_ids(self, sentence: Union[str, list]):
        if not self.parser:
            self.parser = CoreNLPParser(url='http://localhost:9000', tagtype='pos')

        tags = self.convert_sentence_to_tags(sentence)
        ids = self.convert_tags_to_ids(tags)
        print(type(sentence), len(sentence), len(tags), len(ids))
        return list(ids)


if __name__ == "__main__":
    lex_parser = Lex_parser()
    print(list(lex_parser.convert_sentence_to_tags("The price of car is N, "" which is unaffordable. <eos>")))
    print(list(lex_parser.convert_sentence_to_ids("The price of car is N, which is unaffordable.")))
