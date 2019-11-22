import bert_tokenization
import lex_parser


if __name__ == "__main__":
    #sentence = "The price of car is N, which is unaffordable."
    sentence = "aer banknote berlitz calloway centrust cluett fromstein gitano guterman hydro-quebec "

    tokenizer = bert_tokenization.BertTokenizer.from_pretrained('bert-base-uncased')
    parser = lex_parser.Lex_parser()

    bert_tokens = tokenizer.tokenize(sentence)
    print("bert tokens: ", bert_tokens, 'of which the length is', len(bert_tokens))

    bert_token_ids = tokenizer.convert_tokens_to_ids(bert_tokens)
    print("bert token ids:", bert_token_ids, 'of which the length is', len(bert_token_ids))

    lex_tokens = dict(parser.convert_sentence_to_tags(sentence))
    print("lex tags:", lex_tokens, 'of which the length is', len(lex_tokens))

    bert_tokens_tags = tokenizer.convert_tokens_to_tags(bert_tokens, lex_tokens, parser.tag_to_id)
    print("bert token tags", bert_tokens_tags, 'of which the length is', len(bert_tokens_tags))

    bert_token_tag_ids = tokenizer.convert_token_ids_to_tag_ids(bert_token_ids, lex_tokens, parser.tag_to_id)
    print("bert token tag ids", bert_token_tag_ids, 'of which the length is', len(bert_token_tag_ids))