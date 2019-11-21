from nltk.parse.stanford import StanfordParser
# parser = StanfordParser()

from nltk.parse.corenlp import CoreNLPParser


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
    def __init__(self):
        self.tag_to_id = {}
        self.parser = CoreNLPParser(url='http://localhost:9000', tagtype='pos')

    def tokenize(self, sentence):
        return list(self.parser.tokenize(sentence))

    def convert_sentence_to_tags(self, sentence: Union[str, list]):
        if type(sentence) == str:
            sentence = self.parser.tokenize(sentence)

        tags = self.parser.tag(sentence)
        # all_words = set(sentence)
        # res = []

        # for word in sentence:
        #     if word.startswith('['):
        #         print("word:", word)

        for tag in tags:
            # word = tag[0]
            # print("word here", word)
            # if word in all_words or '[' + word + ']' in all_words:
            #     res.append(word)
            if tag[1] not in self.tag_to_id:
                self.tag_to_id[tag[1]] = len(self.tag_to_id)

        # print(self.tag_to_id)
        # print(all_words - set(res))
        # print("to tag ==")
        # print(len(res), res)
        # print(len(all_words))
        # print(len(sentence), sentence)
        # print(len(tags), tags)

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

        if type(sentence) == str:
            sentence = self.parser.tokenize(sentence)

        tags = self.convert_sentence_to_tags(sentence)
        ids = self.convert_tags_to_ids(tags)
        # print(type(sentence), len(sentence), len(tags), len(ids))
        return list(ids)


if __name__ == "__main__":
    lex_parser = Lex_parser()
    print(list(lex_parser.convert_sentence_to_tags("Yesterday, I went to the zoo and saw the tiger. <eos>")))
    print(list(lex_parser.convert_sentence_to_ids("Yesterday, I went to the zoo and saw the tiger.")))
