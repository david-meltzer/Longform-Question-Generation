import os
import wandb
import evaluate
import argparse
import torch
import numpy as np
import pandas as pd
from datasets import load_from_disk

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from transformers.trainer_callback import EarlyStoppingCallback

def parse_args():
    "Overriding default arguments for dataset and training hyperparameters"
    argparser=argparse.ArgumentParser(
        description="process parameters for processing and cleaning dataset."
    )

    argparser.add_argument(
        "--checkpoint",
        type=str,
        default='facebook/bart-base',
        help='model checkpoint used to generate questions from answers'
    )

    argparser.add_argument(
        "--tok",
        type=str,
        default='',
        help='tokenizer checkpoint. Only use if tokenizer is missing \
            from the model repository.'
    )

    argparser.add_argument(
        '--overwrite',
        dest='overwrite',
        default=False,
        action='store_true',
        help='include flag to overwrite existing tokenized dataset files.'
    )

    argparser.add_argument(
        '--max_length',
        type=int,
        default=512,
        help='maximum length of tokenized input for the model.'
    )

    argparser.add_argument(
        '--prompt',
        type=str,
        default='',
        help='additional prompt to add before answer. \
            Only use when training a T5 like model.'
    )

    argparser.add_argument(
        '--fp16',
        dest='fp16',
        default=True,
        action='store_false',
        help='include flag to use float32 precision instead of half-precision.'
    )

    argparser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='size of training and validation batches'
    )

    argparser.add_argument(
        '--num_epochs',
        type=int,
        default=8,
        help='Number of epochs to train the model.'
    )

    argparser.add_argument(
        '--eval_strat',
        type=str,
        default='epoch',
        help='evaluation strategy used by Huggingface trainer'
    )

    argparser.add_argument(
        '--lr',
        type=float,
        default=5.5e-5,
        help='learning rate used to train model.'
    )

    argparser.add_argument(
        '--weight_decay',
        type=float,
        default=0.01,
        help='decoupled weight decay used in Adam optimizer.'
    )

    argparser.add_argument(
        '--save_limit',
        type=int,
        default=3,
        help='limit on how many checkpoints to save during training.'
    )

    argparser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=2,
        help='number of forward passes before taking optimization step'
    )

    argparser.add_argument(
        '--early_stopping_patience',
        type=int,
        default=3,
        help='Stop training model if performance has not improved after \
            this number of epochs or steps (depending on evaluation strategy).'
    )

    return argparser.parse_args()

class Compute_Metrics:
    """
    Defines metrics which are used to monitor the model during training.

    Attributes:
    -----------
    - tokenizer (Huggingface Tokenizer): Pretrained tokenizer corresponding to the model being trained.

    Methods:
    --------
    - compute_metrics: Computes rouge and BERTscore for the model being trained.
    """
    def __init__(self,tokenizer):
        """
        Assigns input tokenizer to be an attribute of the instance.
        """
        self.tokenizer=tokenizer
    
    def compute_metrics(self,eval_pred):
        """
        Computes the rouge and BERTScore for the given label and prediction.

        Input:
        ------
        - eval_pred (tuple): Tuple (predictions,labels) corresponding to predictions of transformer and the associated labels.

        Output:
        -------
        output (dict): Dictionary with keys, value pairs corresponding to the name of the metric and its value on the given batch of data.
        """

        # Load rouge and bertscore metrics from the Huggingface evaluate library.
        rouge=evaluate.load('rouge')
        bertscore=evaluate.load('bertscore')

        #Unpack predictions and labels.
        predictions,labels=eval_pred
        #Decode the predictions and labels from their numerical representation to english.
        decoded_preds=self.tokenizer.batch_decode(predictions)
        labels=np.where(labels!=-100,labels,self.tokenizer.pad_token_id)
        decoded_labels=self.tokenizer.batch_decode(labels,skip_special_tokens=True)
        
        #result is a dictionary with key/value pairs corresponding to the metrics bertscore and rouge and their associated values on the given batch of data.
        result={}
        result['bertscore'] = bertscore.compute(predictions=decoded_preds,
                                            references=decoded_labels,
                                            lang='en')
        
        result['rouge'] = rouge.compute(predictions=decoded_preds,
                                        references=decoded_labels)

        #output is a dictionary with keys 'bertscore' and 'rouge'. In contrast to the result dictionary, for output we average the metric values over the entire batch of data. For 'bertscore' we also drop the key 'hashcode' which we do not use.
        output={}
        for k, met in result.items():
            if met!='hashcode':
                    output[k+'_'+met]=np.mean(result[k][met])
        return output

class Tokenizer_Wrapper:
    """
    Helpful wrapper for Huggingface tokenizers used when training the model.

    Attributes:
    -----------
    - tok (Huggingface Tokenizer): tokenizer associated with the pretrained model.
    - max_length (int): Truncate all input sentences to 512 tokens.

    Methods:
    --------
    - __init__: Assign tokenizer and max_length to attributes of the instance.
    - tokenizer_func: Defines a function which is used to tokenize the answers.text and title columns of the Huggingface dataset.
    """
    def __init__(self,tokenizer,max_length=512):
        """
        Assigns input tokenizer and choice of max_length to attributes of the instance.
        """
        self.tok=tokenizer
        self.max_length=max_length
    
    def tokenizer_func(self,examples):
        """
        Tokenizes the dataset such that the 'answers.text' feature is the input to the transformer and the 'title' feature is the associated label.

        Input:
        -------
        - examples (dict): Dictionary corresponding to a single row of the Huggingface dataset.

        Output:
        -------
        -model_inputs (dict): Dictionary with keys corresponding to the tokenized 'answers.text' column of the original dataset. ALso contains the 'labels' key which corresponds to the tokenized 'title' column of the original dataset.
        """

        #Tokenize and truncate the 'answers.text' feature of the dataset.
        #This will be fed into the encoder part of the transformer architecture.
        model_inputs = self.tok(examples["answers.text"],
                                 max_length=self.max_length,
                                 truncation=True)

        #tokenize and truncate the 'title' feature of the dataset.
        #This will be the sequence the decoder part of transformer attempts to predict.
        labels = self.tok(examples["title"],
                                max_length=self.max_length,
                                truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        
        return model_inputs

class TrainModel:
    """
    Class used to define the model and tokenizer, process the data, and then train the model.

    Attributes:
    -----------
    - dataset (Dataset): Huggingface dataset which has been preprocessed and cleaned to avoid data leakage.
    - tok_datasets (Dataset): Corresponding tokenized dataset.
    - device (str): Device computations will be performed on. To perform computations in a reasonable time need a GPU.
    - fp16 (bool): If true you float16 (half-precision).
    - model_name (str): Name of model being trained. Must be a Seq2Seq model.
    - model (transformer): Instance of the Huggingface transformer defined by 'checkpoint'. 
    - tok (tokenizer): Instance of the corresponding Huggingface tokenizer.
    - data_name (str): Name of folder directory where tokenized data will be saved.
    - data_collator (DataCollatorForSeq2Seq): Huggingface collator designed for sequence-to-sequence transformer models.

    Methods:
    --------
    - __init__: Defines the attributes of a given instance. The sole exception is self.tok_datasets, which is set to None and will be defined using the prepare_data method.
    - prepare_data: Tokenizes dataset and drops columns which are not used during training.
    - train_model: Trains the model using the Seq2Seq Trainer class.

    """
    
    def __init__(self,
                 checkpoint,
                 subreddit='asks',
                 fp16=True):
        """
        Inputs:
        -------
        - checkpoint (str): Model checkpoint on the HuggingfaceHub.
        - subreddit (str): Subreddit to be analyzed. Use 'asks', 'askh', or 'eli5' for the subreddits contained in the ELI5 dataset.
        - fp16 (bool): If True then train the model with half-precision. Else train the model using float32 precision. 
        
        """
        #Location of the cleaned dataset.
        cleaned_file_name=f'./data/{subreddit}_cleaned_data'
        
        #If cleaned file exists then we load the data immediately.
        #Else the process_data file needs to be called.
        if os.path.exists(cleaned_file_name):
            self.dataset=load_from_disk(cleaned_file_name)
        else:
            raise Exception("Dataset does not exist. Run process_data file to create cleaned dataset.")
        #self.tok_datasets will be defined once we run the prepare_data method. 
        self.tok_datasets=None
        
        #Train on a GPU if available.
        self.device= "cuda" if torch.cuda.is_available() else "cpu"
        
        self.fp16=fp16
        self.model_name = checkpoint.split('/')[1]

        #When training a flan-T5 model we need to use full precision.
        if 'flan' in self.model_name.lower():
            self.fp16=False
        
        #Instantiate model and tokenizer using the checkpoint.
        self.model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
        self.tok = AutoTokenizer.from_pretrained(checkpoint)

        self.data_name=subreddit+'_tok_data'
        self.data_collator = DataCollatorForSeq2Seq(self.tok,
                                                    self.model)
    
    def prepare_data(self,
                     max_length=512,
                     prompt='',
                     overwrite=False):
        """
        Inputs:
        -------
        -max_length (int): Truncate all sentences to a max_length number of tokens.
        -prompt (str): Additional to prompt to add before the text of the answer.
        -overwrite (bool): If True then overwrite existing file (if it exists) containing the tokenized data.
        """
    
        #If tokenized data already exists and overwrite is false we load dataset directly from disk.
        if os.path.exists(f'./data/{self.data_name}/{self.model_name}') and not overwrite:
            self.tok_datasets=load_from_disk(f'./data/{self.data_name}/{self.model_name}')
            return

        # If a prompt is given we define a new function add_prompt and apply it to the entire dataset.
        if prompt!='':
            def add_prompt(example):
                example['answers.text']=prompt+example['answers.text']
                return example
            
            self.dataset=self.dataset.map(add_prompt)

        #Define the tokenizer function using the Tokenizer_Wrapper class.
        tok_func=Tokenizer_Wrapper(self.tok,max_length).tokenizer_func
        #tokenize the dataset.
        self.tok_datasets = self.dataset.map(tok_func,batched=True)

        #For training we only need the columns given in the keep_columns list.
        keep_columns=['input_ids','attention_mask','labels']
        drop_cols=[col for col in list(self.dataset['train'].features) \
                   if col not in keep_columns]

        #Drop all extraneous columns and save dataset to disk.
        self.tok_datasets = self.tok_datasets.remove_columns(drop_cols)
        self.tok_datasets.save_to_disk(f'./data/{self.data_name}/{self.model_name}')

        #Log the tokenized data using weights and biases.
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
                    early_stopping_patience=3):
        """
        Train the model using the Seq2SeqTrainer class from the transformers library.

        Inputs:
        -------
        - batch_size (int): Size of training and evaluation batches.
        - num_epochs (int): Number of epochs to train for.
        - eval_strat (str): Evaluation strategy to use during training. Default is 'epoch' so model is evaluated at the end of every epoch. See transformers library for more details.
        - lr (float): Learning rate used to train model using AdamW optimizer.
        - weight_decay (float): Weight decay to apply to all layers except bias and Layernorm layers. Turn off by setting to 0.
        - save_limit (int): Maximum number of models to save locally.
        - gradient_accumulation_steps (int): How many forward passes to run before performing an optimization step.
        - early_stopping_patience (int): Stop training model if performance worsens after this number of evaluation steps.
        """
        #Set up wandb run to log the data.
        with wandb.init(project='Question_Generation',
                 entity=None,
                 job_type='training',
                 name='train_'+self.model_name+'_'+self.data_name) as run:

            #Number of update steps between two logs.
            logging_steps=len(self.tok_datasets['train'])//(2*batch_size)

            #Defines training arguments used in the Seq2SeqTrainer.
            args=Seq2SeqTrainingArguments(
                output_dir= self.model_name+'_'+self.data_name,
                evaluation_strategy=eval_strat,
                save_strategy=eval_strat,
                learning_rate=lr,
                per_device_train_batch_size=batch_size, #We take train and eval batch size to be equal.
                per_device_eval_batch_size=batch_size,
                weight_decay=weight_decay,
                save_total_limit=save_limit,
                num_train_epochs=num_epochs,
                predict_with_generate=True, #Generates tokens which are used to compute the ROUGE and bertscore metrics.
                logging_steps=logging_steps,
                fp16=True if (self.device!='cpu' and self.fp16) else False, #Can only use fp16 if training on a GPU.
                logging_dir=self.model_name+ '_'+self.data_name+'/logs',
                report_to='wandb', #Log data with wandb.
                metric_for_best_model='bertscore_f1', #Use F1 BERTscore metric to determine the best model.
                load_best_model_at_end=True, #Load best performing model at end.
                gradient_accumulation_steps=gradient_accumulation_steps
                )
            
            # Given above training arguments we can now define the trainer.
            trainer = Seq2SeqTrainer(
                self.model,
                args,
                train_dataset=self.tok_datasets['train'],
                eval_dataset=self.tok_datasets['validation'],
                data_collator=self.data_collator,
                tokenizer=self.tok,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=
                                                 early_stopping_patience)],
                compute_metrics=Compute_Metrics(self.tok).compute_metrics #Use ComputeMetrics class to define the compute_metrics function.
                )
            #train and save the model. 
            trainer.train()
            trainer.save_model("./"+self.model_name+'_'+self.data_name)

            #Compute final metrics and log the data on WandB.
            outputs=trainer.evaluate()
            run.log({self.model_name+'_'+self.data_name+"_Performance-data": wandb.Table(dataframe=pd.DataFrame(outputs, index=["Performance"]))})

            #Push final model and tokenizer to hub. Change repo-id to your Huggingface username. 
            self.model.push_to_hub('dhmeltzer/'+self.model_name+'_'+self.data_name)
            self.tok.push_to_hub('dhmeltzer/'+self.model_name+'_'+self.data_name)
            
            #Log the final model on WandB.
            trained_model_art=wandb.Artifact(self.model_name+'_'+self.data_name,type='model')
            trained_model_art.metadata={"hub_id":'dhmeltzer/'+self.model_name+'_'+self.data_name}

            run.log_artifact(trained_model_art)

def complete_train(checkpoint,
                   subreddit,
                   fp16=True,
                   max_length=512,
                   prompt='',
                   overwrite=False,
                   batch_size=4,
                   num_epochs=8,
                   eval_strat='epoch',
                   lr=5.5e-5,
                   weight_decay=0.01,
                   save_limit=3,
                   gradient_accumulation_steps=2,
                   early_stopping_patience=3
                   ):
    """
    complete_train is a wrapper function which takes the TrainModel class, prepares the data, and then trains the model.
    This function is not necessary and one can work with the TrainModel class directly.

    Inputs:
    -------
    - checkpoint (str): Model checkpoint on the HuggingfaceHub.
    - subreddit (str): Subreddit to be analyzed. Use 'asks', 'askh', or 'eli5' for the subreddits contained in the ELI5 dataset.
    - fp16 (bool): If True then train the model with half-precision. Else train the model using float32 precision. 
    - max_length (int): Truncate all sentences to a max_length number of tokens.
    - prompt (str): Additional to prompt to add before the text of the answer.
    - overwrite (bool): If True then overwrite existing file (if it exists) containing the tokenized data.
    - batch_size (int): Size of training and evaluation batches.
    - num_epochs (int): Number of epochs to train for.
    - eval_strat (str): Evaluation strategy to use during training. Default is 'epoch' so model is evaluated at the end of every epoch. See transformers library for more details.
    - lr (float): Learning rate used to train model using AdamW optimizer.
    - weight_decay (float): Weight decay to apply to all layers except bias and Layernorm layers. Turn off by setting to 0.
    - save_limit (int): Maximum number of models to save locally.
    - gradient_accumulation_steps (int): How many forward passes to run before performing an optimization step.
    - early_stopping_patience (int): Stop training model if performance worsens after this number of evaluation steps.
    """

    train_model=TrainModel(checkpoint,
                           subreddit=subreddit,
                           fp16=fp16)

    train_model.prepare_data(max_length=max_length,
                             prompt=prompt,
                             overwrite=overwrite
                             )
    
    train_model.train_model(batch_size=batch_size,
                            num_epochs=num_epochs,
                            eval_strat=eval_strat,
                            lr=lr,
                            weight_decay=weight_decay,
                            save_limit=save_limit,
                            gradient_accumulation_steps=gradient_accumulation_steps,
                            early_stopping_patience=early_stopping_patience)

if __name__ == "__main__":
    args=vars(parse_args())

    complete_train(checkpoint=args.checkpoint,
                   subreddit=args.subreddit,
                   fp16=args.fp16,
                   max_length=args.max_length,
                   prompt=args.prompt,
                   overwrite=args.overwrite,
                   batch_size=args.batch_size,
                   num_epochs=args.num_epochs,
                   eval_strat=rgs.eval_strat,
                   lr=args.lr,
                   weight_decay=args.weight_decay,
                   save_limit=args.save_limit,
                   gradient_accumulation_steps=args.gradient_accumulation_steps,
                   early_stopping_patience=args.early_stopping_patience,
                   )