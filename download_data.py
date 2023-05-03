import wandb
import argparse
import os
from datasets import load_from_disk, load_dataset,DatasetDict

def parse_args():
    "Overriding default arguments for downloading data"
    argparser = argparse.ArgumentParser(
        description='Process parameters for downloading Huggingface dataset.'
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
        "--local_file",
        type=str,
        default='',
        help='local file containing Huggingface dataset.'
    )
    
    return argparser.parse_args()

def download_raw_data(subreddit='asks',
                      overwrite = False):  
    
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
    args = vars(parse_args())
    subreddit = args.subreddit
    overwrite = args.overwrite
    local_file = args.local_file

    download_raw_data(subreddit=subreddit,
                      overwrite = overwrite,
                      local_file=local_file)