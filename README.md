# Integrating Grammar Tree Structure Into BERT Language Model

## Author
Shiyi Han, Yiming Zhang, Shunjia Zhu

## Run on SQuAD
1. Clone the repo
2. Install the requried package. Use 'pip install <package>'
2. Download the stanford nlp parser:

```cd Enbert/```

```wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip --no-check-certificate```

```unzip stanford-corenlp-full-2018-10-05.zip```

```cd stanford-corenlp-full-2018-10-05```

3. Preprocess the data:
    1. Start the lex parser, cd the folder and 
```java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos,lemma,ner,parse,depparse -status_port 9000 -port 9000 -timeout 15000 &```
    2. python squad_data_process.py

4. Execute run_squad.py
Testing:

```python run_squad.py --bert_model bert-base-uncased --output_dir /path/to/small_param_model/ --do_predict --predict_file /path/to/EnBert/data/squad/dev-v2.0.json --version_2_with_negative --fp16```

Evaluaiton: 

```python eval.script.py EnBert/data/squad/dev-v2.0.json small_param_model/predictions.json```

# Run on GLUE
1. Download the stanford nlp parser:

```cd Enbert/```

```wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip```

```unzip stanford-corenlp-full-2018-10-05.zip```

```cd stanford-corenlp-full-2018-10-05```

2. Download the glue data:

```python download_glue_data.py --data_dir ../glue_data --tasks all```

3. Preprocess the data:
    1. start the lex parser, cd the folder and 
```java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos,lemma,ner,parse,depparse -status_port 9000 -port 9000 -timeout 15000 &```
    
4. Execute run_glue.py and specify the task
```sh run_glue.sh MRPC POS```
