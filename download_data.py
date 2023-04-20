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
                      overwrite = False,
                      local_file = ''):
    """
    Downloads raw data from either a local file containing the Huggingface dataset or from the ELI5 dataset directly.

    Inputs:
    -------
    - subreddit (str): Name of subreddit to download question/answer pairs from. Choose 'asks', 'askh', or 'eli5' to download the 'askscience', 'askhistorians' or 'eli5' splits from the ELI5 dataset. For any other choice of subreddit the file containing the raw data must be provided explicitly in the local_file.
    - overwrite (bool): Set to True to redownload data and overwrite existing files. 
    - local_file (str): Location of the raw data on the local system. Only needed if subreddit is not 'asks', 'askh', or 'eli5'.

    Output:
    - dataset (Dataset): Huggingface dataset containing the raw data.
    
    """

    #Make data directory if it does not exist already.
    if not os.path.exists('./data'):
            os.mkdir('./data')

    #If local_file is given and exists locally the dataset is directly loaded from there. 
    if local_file is not '' and os.path.exists(local_file):
        dataset=load_from_disk(local_file)
    else:
        #Define default raw_file_name if local_file is not given or does not exist.
        raw_file_name = f'./data/{subreddit}_raw_data'
        print(f'local_file does not exist. Attempting download with default filename: {raw_file_name}.')
        #Load dataset if raw_file_name already exists and overwrite is false.
        if os.path.exists(raw_file_name) and not overwrite:
            dataset=load_from_disk(raw_file_name)
        #If local file does not exist and subreddit is not in ELI5 dataset we raise an exception.
        elif subreddit not in ['asks','askh','eli5']:
             raise Exception(f'Default filename {raw_file_name} for subreddit {subreddit} not found. File needs to be either created manually or choose subreddit to be "asks", "askh", or "eli5" to download corresponding subreddit data from the ELI5 dataset, see https://huggingface.co/datasets/eli5.')
        else:
            # Load full ELI5 dataset.
            dataset_eli5 = load_dataset('eli5')
            
            #Define new dataset just containing the splits for the given subreddit.
            dataset = DatasetDict()
            dataset['train'] = dataset_eli5['train_'+subreddit]
            dataset['validation'] = dataset_eli5['validation_'+subreddit]
            dataset['test'] = dataset_eli5['test_'+subreddit]
            
            #Flatten dataset so answer text, answer URL, and answer ID are given separate columns.
            #dataset.flatten also produces corresponding columns for the post and title.
            dataset = dataset.flatten()

        dataset.save_to_disk(raw_file_name)
        
        #Use wandb to log the raw data.
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

    download_raw_data(subreddit=args.subreddit,
                      overwrite = args.overwrite,
                      local_file=args.local_file)