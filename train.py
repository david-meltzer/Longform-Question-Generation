import os
import wandb
import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from transformers.trainer_callback import EarlyStoppingCallback



class Compute_Metrics:
    def __init__(self,tokenizer):
        self.tokenizer=tokenizer
    
    def compute_metrics(self,eval_pred):
        rouge=evaluate.load('rouge')
        bertscore=evaluate.load('bertscore')

        predictions,labels=eval_pred
        decoded_preds=self.tokenizer.batch_decode(predictions)
        labels=np.where(labels!=-100,labels,self.tokenizer.pad_token_id)
        decoded_labels=self.tokenizer.batch_decode(labels,skip_special_tokens=True)
        
        result={}

        result['bertscore'] = bertscore.compute(predictions=decoded_preds,
                                            references=decoded_labels,
                                            lang='en')
        
        result['rouge'] = rouge.compute(predictions=decoded_preds,
                                references=decoded_labels)

        output={}
        for k in result:
            for met in result[k]:
                if met!='hashcode':
                    output[k+'_'+met]=np.mean(result[k][met])

        return output

class Tokenizer_Wrapper:
    def __init__(self,tokenizer,max_length=512):
        self.tok=tokenizer
        self.max_length=max_length
    
    def tokenizer_func(self,examples):
        model_inputs = self.tok(examples["answers.text"],
                                 max_length=self.max_length,
                                 truncation=True)

        labels = self.tok(examples["title"],
                                max_length=self.max_length,
                                truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        
        return model_inputs

class TrainModel:
    
    def __init__(self,
                 checkpoint,
                 dataset,
                 subreddit='asks',
                 tok=None,
                 fp16=True):
        
        self.device= "cuda" if torch.cuda.is_available() else "cpu"
        
        self.fp16=fp16
        self.model_name = checkpoint.split('/')[1]

        if 'flan' in self.model_name.lower():
            self.fp16=False
        
        self.model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
        
        if tok is None:
            self.tok = AutoTokenizer.from_pretrained(checkpoint)
        else:
            self.tok = tok
        self.data_name = subreddit+'_tok_data'
        self.dataset=dataset
        self.data_collator = DataCollatorForSeq2Seq(self.tok, 
                                                    self.model)
    
    def prepare_data(self,
                     max_length=512,
                     prompt='',
                     overwrite=False):
    
        if os.path.exists(f'./data/{self.data_name}/{self.model_name}') and not overwrite:
            self.tok_datasets=load_from_disk(f'./data/{self.data_name}/{self.model_name}')
            return

        if prompt!='':
            def add_prompt(example):
                example['answers.text']=prompt+example['answers.text']
                return example
            
            self.dataset=self.dataset.map(add_prompt)
            

        tok_func=Tokenizer_Wrapper(self.tok,max_length).tokenizer_func
        self.tok_datasets = self.dataset.map(tok_func,batched=True)

        keep_columns=['input_ids','attention_mask','labels']
        drop_cols=[col for col in list(self.dataset['train'].features) \
                   if col not in keep_columns]

        self.tok_datasets = self.tok_datasets.remove_columns(drop_cols)
        self.tok_datasets.save_to_disk(f'./data/{self.data_name}/{self.model_name}')

        with wandb.init(project='Question_Generation', 
                entity = None, 
                job_type = 'logging_tokenized_data',
                name = 'tok_'+self.model_name+'_'+self.data_name) as run:

            tok_data_art=wandb.Artifact(self.data_name+'_'+self.model_name,type='dataset')
            tok_data_art.add_dir(f'./data/{self.data_name}/{self.model_name}')
    
            run.log_artifact(tok_data_art)
        
    def train_model(self,batch_size=4,num_epochs=8,eval_strat='epoch',
                    lr=5.5e-5,weight_decay=0.01,save_limit=3,
                    gradient_accumulation_steps=2,
                    early_stopping_patience=3,
                    load_file='',local_files_only=False):
        
        with wandb.init(project='Question_Generation', 
                 entity=None, 
                 job_type='training',
                 name='train_'+self.model_name+'_'+self.data_name) as run:
            
            if load_file!='':
                self.model=AutoModelForSeq2SeqLM.from_pretrained(load_file,
                                                                 local_files_only=local_files_only)

            logging_steps=len(self.tok_datasets['train'])//(2*batch_size)

            args=Seq2SeqTrainingArguments(
                output_dir= self.model_name+'_'+self.data_name,
                evaluation_strategy=eval_strat,
                save_strategy=eval_strat,
                learning_rate=lr,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                weight_decay=weight_decay,
                save_total_limit=save_limit,
                num_train_epochs=num_epochs,
                predict_with_generate=True,
                logging_steps=logging_steps,
                fp16=True if (self.device!='cpu' and self.fp16) else False,
                logging_dir=self.model_name+ '_'+self.data_name+'/logs',
                report_to='wandb',
                metric_for_best_model='bertscore_f1',
                load_best_model_at_end=True,
                gradient_accumulation_steps=gradient_accumulation_steps
                )
            
            trainer = Seq2SeqTrainer(
                self.model,
                args,
                train_dataset=self.tok_datasets['train'],
                eval_dataset=self.tok_datasets['validation'],
                data_collator=self.data_collator,
                tokenizer=self.tok,
                callbacks=[EarlyStoppingCallback(\
                                                 early_stopping_patience=\
                                                 early_stopping_patience)],
                compute_metrics=Compute_Metrics(self.tok).compute_metrics
                )
            
            trainer.train()

            outputs=trainer.evaluate()
            trainer.save_model("./"+self.model_name+'_'+self.data_name)
            
            run.log({self.model_name+'_'+self.data_name+"_Performance-data": wandb.Table(dataframe=pd.DataFrame(outputs, index=["Performance"]))})
            self.model.push_to_hub('dhmeltzer/'+self.model_name+'_'+self.data_name)
            self.tok.push_to_hub('dhmeltzer/'+self.model_name+'_'+self.data_name)
            
            trained_model_art=wandb.Artifact(self.model_name+'_'+self.data_name,type='model')
            trained_model_art.metadata={"hub_id":'dhmeltzer/'+self.model_name+'_'+self.data_name}

if __name__ == "__main__":
    