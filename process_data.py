import wandb
import argparse
import re
import os
import torch
import numpy as np
from datasets import load_from_disk
from download_data import download_raw_data
from sentence_transformers import SentenceTransformer


def parse_args():
    "Overriding default arguments for processing data"
    argparser=argparse.ArgumentParser(
        description="process parameters for processing and cleaning dataset."
    )
    
    argparser.add_argument(
        "--subreddit",
        type=str,
        default='asks',
        help='subreddit to use from ELI5 dataset. Options are \
            "asks" "askh" and "eli5" for askscience, askhistory, and ELI5 subreddits.'
    )
    
    argparser.add_argument(
        '--overwrite',
        dest='overwrite',
        default=False,
        action='store_true',
        help='include flag to overwrite existing dataset files.'
    )

    argparser.add_argument(
        "--sent_model",
        type=str,
        default='paraphrase-MiniLM-L6-v2',
        help='checkpoint for SentenceTransformer used to embed answers.'
    )

    argparser.add_argument(
        "--chunk_length",
        type=int,
        default=128,
        help='Split answers into substrings of size "chunk_length".\
            Each substring is embedded using sentence transformer and we \
                sum all subtrings to get embedding of entire answer.'
    )

    argparser.add_argument(
        "--cutoff",
        type=float,
        default=.9,
        help="If cosine distance between two answers in the train/validation/test \
            splits is greater than cutoff, answer is removed from one of the datasets \
                to avoid data leakage."
    )

    argparser.add_argument(
        "--min_sent_length",
        type=int,
        default=20,
        help= "Only keep answers which are at least 'min_sent_length' characters long."
    )

    return argparser.parse_args()


def preprocess_func(example):
    example['answers.text']=example['answers.text'][0]
    example['answers.text']=re.sub('>.*?\n',' ',example['answers.text'])
    example['answers.text']=' '.join(example['answers.text'].lower().split())

    example['answers.a_id']=example['answers.a_id'][0]
    example['answers.score']=example['answers.score'][0]

    example['title']=' '.join(example['title'].lower().split())
    example['selftext']=' '.join(example['selftext'].lower().split())

    return example

def preprocess_data(dataset):
    dataset=dataset.map(preprocess_func)
    return dataset

def log_processed_data(subreddit='asks',
                       overwrite=False,
                       min_sent_length=20):
    
    raw_file_name = f'./data/{subreddit}_raw_data'
    processed_file_name=f'./data/{subreddit}_processed_data'
    
    if os.path.exists(processed_file_name) and not overwrite:
        return load_from_disk(processed_file_name)
    
    if os.path.exists(raw_file_name) and not overwrite:
        dataset=load_from_disk(raw_file_name)
    else:
        dataset=download_raw_data(subreddit=subreddit)

    drop_cols=['subreddit','document','answers.a_id','q_id']
    
    
    ds_reduced = dataset.remove_columns(drop_cols)
    
    ds_reduced = ds_reduced.filter(lambda x:\
                                    len(x['answers.text'].split())>min_sent_length)
    
    ds_reduced = ds_reduced.filter(lambda x:\
                                    'ask anything wednesday' not in x['title'])
    
    ds_reduced.save_to_disk(processed_file_name)
    
    with wandb.init(project='Question_Generation', 
                entity = None, 
                job_type = 'logging_processed_data',
                name = 'processed_data') as run:
        
        proc_data_art=wandb.Artifact(subreddit+'_processed_data',type='dataset')
        proc_data_art.add_dir(processed_file_name)
        
        run.log_artifact(proc_data_art)

    return ds_reduced

def par_to_vec(model,sent,chunk_length=128):
    
    chunks=[sent[0+i:chunk_length+i] 
            for i in range(0,len(sent),chunk_length)]
    
    embeddings=model.encode(chunks)
    return np.sum(embeddings,axis=0,keepdims=True)

def dataset_par_to_vec(model,example,chunk_length=128):
    example['sent_vec']=par_to_vec(model,
                                   example['answers.text'],
                                   chunk_length=chunk_length)
    return example

def clean_and_embed_data(sent_model,
                         subreddit='asks',
                         overwrite=False,
                         cutoff=.9,
                         min_sent_length=20):
    
    processed_file_name=f'./data/{subreddit}_processed_data'
    cleaned_file_name=f'./data/{subreddit}_cleaned_data'

    if os.path.exists(cleaned_file_name) and not overwrite:
        return load_from_disk(cleaned_file_name)
    
    if os.path.exists(processed_file_name):
        ds_reduced = load_from_disk(processed_file_name)
    else:
        ds_reduced = log_processed_data(subreddit='asks',
                       overwrite=overwrite,
                       min_sent_length=min_sent_length)
    
    ds_reduced_emb=ds_reduced.map(lambda x: dataset_par_to_vec(sent_model,x))
    ds_reduced_emb.set_format('torch')
    
    train_vecs=ds_reduced_emb['train']['sent_vec']
    valid_vecs=ds_reduced_emb['validation']['sent_vec']
    test_vecs=ds_reduced_emb['test']['sent_vec']

    norm_train=torch.sqrt(torch.sum(train_vecs**2,axis=1,keepdims=True))
    norm_valid=torch.sqrt(torch.sum(valid_vecs**2,axis=1,keepdims=True))
    norm_test=torch.sqrt(torch.sum(test_vecs**2,axis=1,keepdims=True))

    valid_test=torch.matmul(test_vecs/norm_test,
          torch.transpose(valid_vecs/norm_valid,0,1))

    train_test=torch.matmul(test_vecs/norm_test,
          torch.transpose(train_vecs/norm_train,0,1))

    train_valid=torch.matmul(valid_vecs/norm_valid,
          torch.transpose(train_vecs/norm_train,0,1))
    
    sim={}
    sim['train','test']=torch.where(train_test>cutoff)
    sim['valid','test']=torch.where(valid_test>cutoff)
    sim['train','valid']=torch.where(train_valid>cutoff)

    train_rem_idxs = np.concatenate((sim['train','test'][0].numpy(),
                                     sim['train','valid'][0].numpy()))
    
    train_rem_idxs = set(train_rem_idxs)

    valid_rem_idxs = np.concatenate((sim['train','test'][0].numpy(),
                                     sim['train','valid'][0].numpy()))
    
    valid_rem_idxs = set(valid_rem_idxs)

    ds_reduced_emb['train']=ds_reduced_emb['train'].filter(lambda _,idx:idx 
                                   not in train_rem_idxs,with_indices=True)

    ds_reduced_emb['validation']=ds_reduced_emb['validation'].filter(lambda _,idx:idx 
                                   not in valid_rem_idxs,with_indices=True)
    
    for split in ['train','validation','test']:
        removed_sent="your submission has been removed"

        ds_reduced_emb[split]=ds_reduced_emb[split].filter(lambda example:
                                                       removed_sent not in example['answers.text'])
    
    ds_reduced_emb.save_to_disk(cleaned_file_name)

    with wandb.init(project='Question_Generation', 
            entity = None, 
            job_type = 'logging_cleaned_data',
            name = 'cleaned_data') as run:
    
        cleaned_data_art=wandb.Artifact(subreddit+'_cleaned_data',type='dataset')
        cleaned_data_art.add_dir(cleaned_file_name)
        
        run.log_artifact(cleaned_data_art)

    return ds_reduced_emb

if __name__ == "__main__":
    
    args=vars(parse_args())

    sent_model_checkpoint=args.sent_model
    sent_model=SentenceTransformer(sent_model_checkpoint)

    clean_and_embed_data(sent_model,
                         subreddit=args.subreddit,
                         overwrite=args.overwrite,
                         cutoff=args.cutoff,
                         min_sent_length=args.min_sent_length)