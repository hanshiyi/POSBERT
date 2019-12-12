# Integrating Grammar Tree Structure Into BERT Language Model

## Author
Shiyi Han, Yiming Zhang, Shunjia Zhu

## Run on SQuAD
1. Clone the repo
2. Install the requried package. Use 'pip install <package>'
3. Download and unzip SQuAD datasets. 
3. Download the stanford nlp parser:

    ```cd POSBERT/
    
    wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip --no-check-certificate
    
    unzip stanford-corenlp-full-2018-10-05.zip
    
    cd stanford-corenlp-full-2018-10-05
    ```

4.  Start the lex parser.
    ```
    cd stanford-corenlp-full-2018-10-05
    
    java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos,lemma,ner,parse,depparse -status_port 9000 -port 9000 -timeout 15000 &
    ```
5. Preprocess the data. 
    ```
    python squad_data_process.py
    ```

6. Execute run_squad.py
- Testing:
    ```
    python run_squad.py --bert_model bert-base-uncased --output_dir /path/to/small_param_model/ --do_predict --predict_file /path/to/data/squad/dev-v2.0.json --version_2_with_negative --fp16
    ```

- Evaluaiton: 

    ```
    python eval.script.py /path/to/data/squad/dev-v2.0.json small_param_model/predictions.json
    ```

Our result:

SQuAD 1.1 results on Dev set

|  Model  |   F1  | Exact Match |
|:-------:|:-----:|:-----------:|
|   BERT  | 82.27 |    73.59    |
| POSBERT | 88.23 |    80.44    |

SQuAD 2.0 results on Dev set

|  Model  |   F1  | Exact Match |
|:-------:|:-----:|:-----------:|
|   BERT  | 66.92 |    63.48    |
| POSBERT | 75.67 |    72.12    |


## Run on GLUE
1. Download the stanford nlp parser:
    
    ```
    cd POSBERT/
    
    wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip
    
    unzip stanford-corenlp-full-2018-10-05.zip
    
    cd stanford-corenlp-full-2018-10-05
    ```

2. Download GLUE data:

    ```
    python download_glue_data.py --data_dir ../glue_data --tasks all
    ```

3. Preprocess the data. Start the lex parser.
    ```
    java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos,lemma,ner,parse,depparse -status_port 9000 -port 9000 -timeout 15000 &
    ```
    
4. Execute run_glue.py and specify the task
    ```
    python run_glue.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --task_name MRPC \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir /path/to/glue_data/MRPC \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=8   \
    --per_gpu_train_batch_size=8   \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir temp/MRPC\
    --overwrite_output_dir
    ```