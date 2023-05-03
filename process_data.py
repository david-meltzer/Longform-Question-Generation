import argparse
import wandb
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
    """
    Used to preprocess a Huggingface dataset for a given subreddit.
    Preprocessing only keeps the top answer (corresponding metadata) and lower-cases text.
    To prevent dataleakage, quotes of the original question are removed from the top answer.

    Inputs:
    -------
    - example (dict): Dictionary corresponding to one row in the dataset

    Output:
    -------
    - example (dict): Dictionary corresponding to transformed row.
    """

    #keep top answer.
    example['answers.text']=example['answers.text'][0]
    #Remove quotation of the original title and post from the answer.
    #Used to avoid data leakage.
    example['answers.text']=re.sub('>.*?\n',' ',example['answers.text'])
    #Lower case all text.
    example['answers.text']=' '.join(example['answers.text'].lower().split())

    #Keep only the top answer id and answer score.
    example['answers.a_id']=example['answers.a_id'][0]
    example['answers.score']=example['answers.score'][0]

    #Lower case the title and original post.
    example['title']=' '.join(example['title'].lower().split())
    example['selftext']=' '.join(example['selftext'].lower().split())

    return example

def preprocess_data(dataset):
    """
    Helper function to apply "preprocess_func" to the entire dataset.

    Inputs:
    -------
    - dataset (Dataset): Huggingface dataset to be processed.

    Outputs:
    --------
    - dataset (Dataset): Dataset transformed by "preprocess_func" function.
    """
    dataset=dataset.map(preprocess_func)
    return dataset

def log_processed_data(subreddit='asks',
                       overwrite=False,
                       min_sent_length=20):
    """"
    Preprocesses Huggingface dataset and logs the new dataset in WandB.
    
    Inputs:
    -------
    - subreddit (str): String corresponding to the subreddit. To use ELI5 dataset options are 
                     'asks', 'askh', or 'eli5'. 
    - overwrite (bool): If True than reapply all preprocessing steps and overwrite existing files.
    - min_sent_length (int): Only keep posts where the top answer has length>min_sent_length.
                             Used to avoid including answers which are short and uninformative.
    
    Outputs:
    --------
    - ds_reduced (Dataset): Dataset after all preprocessing steps.

    """
    #File containing raw data.
    raw_file_name = f'./data/{subreddit}_raw_data'

    #File containing processed data.
    processed_file_name=f'./data/{subreddit}_processed_data'

    #If processed file exists and overwrite=False we return existing dataset.
    if os.path.exists(processed_file_name) and not overwrite:
        return load_from_disk(processed_file_name)

    #If raw file exists and overwrite=False we load raw data.
    #If not, then we download raw data.
    if os.path.exists(raw_file_name) and not overwrite:
        dataset=load_from_disk(raw_file_name)
    else:
        dataset=download_raw_data(subreddit=subreddit)

    #Drop the following columns
    drop_cols=['subreddit','document','answers.a_id','q_id']
    ds_reduced = dataset.remove_columns(drop_cols)

    #Only keep posts where the answer length is >min_sent_length
    ds_reduced = ds_reduced.filter(lambda x:\
                                    len(x['answers.text'].split())>min_sent_length)
    
    #Remove posts where post title contains 'ask anything wednesday'.
    #These are not useful because the post title does not contain a question!
    ds_reduced = ds_reduced.filter(lambda x:\
                                    'ask anything wednesday' not in x['title'])
    
    #Save processed data to disk.
    ds_reduced.save_to_disk(processed_file_name)
    
    #Launch wandb process which logs the processed data.
    #Note: project and entity can be changed depending on your preference.
    with wandb.init(project='Question_Generation', 
                entity = None, 
                job_type = 'logging_processed_data',
                name = 'processed_data') as run:
        
        #Define new artifact for processed data.
        proc_data_art=wandb.Artifact(subreddit+'_processed_data',type='dataset')
        #Add artifact to WandB project
        proc_data_art.add_dir(processed_file_name)
        #Log the artifact on WandB.
        run.log_artifact(proc_data_art)

    return ds_reduced

def par_to_vec(model,sent,chunk_length=128):
    """
    Perform paragraph embedding. Entire answer post is embedded in a high-dimensional vector space using a transformer model.

    Inputs:
    -------
    - model (transformer): Huggingface transformer used to encode the paragraph. Typically chosen to be a sentence-transformer model.
    - sent (str): Sentence or paragraph to be embedded.
    - chunk_length (int): Break up sent into chunks of size chunk-length, encode each one using transformer model, and  then average over all chunks.

    Outputs:
    --------
    - avg_embedding (np.array): Embedding of sent into higher-dimensional vector space.
    """
    
    #Break up sent into strings of length chunk_length.
    chunks=[sent[0+i:chunk_length+i] 
            for i in range(0,len(sent),chunk_length)]
    
    #apply model to each chunk individually.
    embeddings=model.encode(chunks)
    
    #Average embeddings for each chunk to obtain embedding for sent.
    avg_embedding=np.sum(embeddings,axis=0,keepdims=True)
    return avg_embedding

def dataset_par_to_vec(model,example,chunk_length=128):
    """
    Helper function used to apply function par_to_vec to a Huggingface dataset.

    Input:
    -------
    - example (dict): Single row of a Huggingface dataset.

    Output:
    - example (dict): Same row with 'sent_vec' key added.
    """

    #Include new key 'sent_vec' corresponding to embedding of the top answer.
    example['sent_vec']=par_to_vec(model,
                                   example['answers.text'],
                                   chunk_length=chunk_length)
    return example

def clean_and_embed_data(model,
                         subreddit='asks',
                         overwrite=False,
                         cutoff=.9,
                         min_sent_length=20,
                         chunk_length=128):
    """"
    Uses embeddings of answers generated by transformer model to check and remove any data leakage between the train, validation, and test splits of the dataset. Logs cleaned dataset using WandB.

    Inputs:
    -------
    - model (transformer): Sentence-transformer model used to perform embeddings.
    - subreddit (str): Subreddit to analyze. For ELI5 dataset can be 'asks', 'askh', or 'eli5'.
    - overwrite (bool): If set to True overwrite all existing datasets.
    - cutoff (float): If cosine-similarity between different answers in the train/validation/test splits are greater than cutoff we remove the post from the train and/or validation dataset.
    - min_sent_length (int): Only keep posts where the answer length > min_sent_length. 
    """
    
    # Name of file containing processed data.
    processed_file_name=f'./data/{subreddit}_processed_data'
    # Name of file contained embedded and cleaned data.
    cleaned_file_name=f'./data/{subreddit}_cleaned_data'

    # If cleaned file already exists we load and return existing cleaned dataset.
    if os.path.exists(cleaned_file_name) and not overwrite:
        return load_from_disk(cleaned_file_name)
    
    # If processed file already exists we load processed dataset from there.
    # Else we call the "log_processed_data" function.
    if os.path.exists(processed_file_name):
        ds_reduced = load_from_disk(processed_file_name)
    else:
        ds_reduced = log_processed_data(subreddit='asks',
                       overwrite=overwrite,
                       min_sent_length=min_sent_length)
    
    #Embed answers using model.
    ds_reduced_emb=ds_reduced.map(lambda x: dataset_par_to_vec(model,x,chunk_length=chunk_length))
    #Change format to torch for later matrix multiplications.
    ds_reduced_emb.set_format('torch')
    
    #Extract embedding vectors for the train, validation, and test splits.
    train_vecs=ds_reduced_emb['train']['sent_vec']
    valid_vecs=ds_reduced_emb['validation']['sent_vec']
    test_vecs=ds_reduced_emb['test']['sent_vec']

    # Compute the norms of the embedding vectors for the train, validation and test splits.
    norm_train=torch.sqrt(torch.sum(train_vecs**2,axis=1,keepdims=True))
    norm_valid=torch.sqrt(torch.sum(valid_vecs**2,axis=1,keepdims=True))
    norm_test=torch.sqrt(torch.sum(test_vecs**2,axis=1,keepdims=True))

    # valid_test computes the cosine-similarity between the test and validation vectors for all possible combinations.
    # train_test and train_valid are the same, but compute cosine-similarity bet
    valid_test=torch.matmul(test_vecs/norm_test,
          torch.transpose(valid_vecs/norm_valid,0,1))

    train_test=torch.matmul(test_vecs/norm_test,
          torch.transpose(train_vecs/norm_train,0,1))

    train_valid=torch.matmul(valid_vecs/norm_valid,
          torch.transpose(train_vecs/norm_train,0,1))
    
    # sim is a dictionary which contains the indices where elements from train/validation/test sets have a cosine similarity greater than the cutoff, i.e. the elements are too similar.
    sim={}
    sim['train','test']=torch.where(train_test>cutoff)
    sim['valid','test']=torch.where(valid_test>cutoff)
    sim['train','valid']=torch.where(train_valid>cutoff)

    #train_rem_idxs contains all the elements from the training set we remove to avoid data leakeage.
    train_rem_idxs = np.concatenate((sim['train','test'][0].numpy(),
                                     sim['train','valid'][0].numpy()))
    #Convert train_rem_idxs to a set to avoid duplication.
    train_rem_idxs = set(train_rem_idxs)
    #Same as above but now for the validation set.
    #We never remove elements from the test set.
    valid_rem_idxs = np.concatenate((sim['train','test'][0].numpy(),
                                     sim['train','valid'][0].numpy()))
    
    valid_rem_idxs = set(valid_rem_idxs)

    #Filter the train and validation set to remove possible data leakage.
    ds_reduced_emb['train']=ds_reduced_emb['train'].filter(lambda _,idx:idx 
                                   not in train_rem_idxs,with_indices=True)

    ds_reduced_emb['validation']=ds_reduced_emb['validation'].filter(lambda _,idx:idx 
                                   not in valid_rem_idxs,with_indices=True)
    
    #Remove answer posts which have been removed by a reddit admin or bot.
    removed_sent="your submission has been removed"
    for split in ['train','validation','test']:   
        ds_reduced_emb[split]=ds_reduced_emb[split].filter(lambda example:
                                                       removed_sent not in example['answers.text'])
    
    #Save final result to disk
    ds_reduced_emb.save_to_disk(cleaned_file_name)
    #Start wandb run which will log the cleaned data.
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
    #Sentence model checkpoint.
    sent_model_checkpoint=args.sent_model
    #Instantiation of the sentence model.
    sent_model=SentenceTransformer(sent_model_checkpoint)

    clean_and_embed_data(sent_model,
                         subreddit=args.subreddit,
                         overwrite=args.overwrite,
                         cutoff=args.cutoff,
<<<<<<< HEAD
                         min_sent_length=args.min_sent_length)
=======
                         min_sent_length=args.min_sent_length,
                         chunk_length=args.chunk_length)
>>>>>>> bcff9773844697770c9b4a4910f235e0ae815bbf
