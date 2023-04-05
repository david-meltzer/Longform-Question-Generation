import wandb
import os
from datasets import load_from_disk, load_dataset,DatasetDict


def download_raw_data(subreddit='asks',overwrite = False):
    
    raw_file_name = f'./data/{subreddit}_raw_data'

    if os.path.exists(raw_file_name) and not overwrite:
        dataset=load_from_disk(raw_file_name)

    else:
        if not os.path.exists('./data'):
            os.makedir('./data')

        dataset_eli5 = load_dataset('eli5')
        
        dataset = DatasetDict()
        dataset['train'] = dataset_eli5['train_'+subreddit]
        dataset['validation'] = dataset_eli5['validation_'+subreddit]
        dataset['test'] = dataset_eli5['test_'+subreddit]
        
        dataset = dataset.flatten()

        dataset.save_to_disk(raw_file_name)
        
        with wandb.init(project='Question_Generation', 
                 entity=None, 
                 job_type='logging_data',
                 name='logging_data') as run:    
            
           
            raw_data_art=wandb.Artifact(subreddit+'_raw_data','dataset')
            raw_data_art.add_dir(raw_file_name)
            run.log_artifact(raw_data_art)

    return dataset

if __name__ == "__main__":