# EnBert
## Update(11/18):
we will use modules in the repo https://github.com/jurekkow/bert-squad-demo to 
parse squad dataset.

Usage: please refer to main() in run_squad.py

## Update(12/4)
Add glue data, Please use the following command to download glue data,

```python download_glue_data.py --data_dir data/glue --tasks all```


Then we use the modules in the repo https://github.com/huggingface/transformers to parse 
the glue dataset.


# run the code step by step
1. clone the repo
2. download the stanford nlp parser:

```cd Enbert/```

```wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip```

```unzip stanford-corenlp-full-2018-10-05.zip```

```cd stanford-corenlp-full-2018-10-05```

3. download the glue data:

```python download_glue_data.py --data_dir data/glue --tasks all```

4. preprocess the data:
    1. start the lex parser, cd the folder and 
```java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos,lemma,ner,parse,depparse -status_port 9000 -port 9000 -timeout 15000 &```
  
    2. process the squad data
    
    3. process the glue data
    ```
   python run_glue.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --task_name MRPC \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir data/glue/MRPC \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=8   \
    --per_gpu_train_batch_size=8   \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir temp/MRPC\
    --overwrite_output_dir
   ```