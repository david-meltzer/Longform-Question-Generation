# Question Generation for Scientific Text

Can we teach large language models (LLMs) to ask scientific questions? 
In this project we study to what extent we train to read in a span of text on a scientific subject and then produce an interesting and coherent question which is also answerable from the original text. 
The inspiration for this project came from this interesting <a href="https://arxiv.org/abs/2210.11536">paper</a> which studied how to build a language model to generate questions based on news articles.
We will study how some of the ideas introduced in that paper can also be used in the purely scientific domain by training large language models on the r/AskScience dataset.

All our fine-tuned models are hosted on Huggingface, see any model <a href="https://huggingface.co/dhmeltzer">here<a> which contains the text "asks" or "askscience" in the model name. The best performing model is a fine-tuned BART-large model which can be tested <a href="https://huggingface.co/dhmeltzer/bart-large_askscience-qg">here</a>. Models with the word "yake" in the title require that any input text is modified by adding additional keywords which we extracted using the <a href="https://github.com/LIAAD/yake">YAKE</a> model. The (cleaned) dataset can be found <a href="https://huggingface.co/datasets/dhmeltzer/ask-science-qg">here</a> the same dataset with keywords added can be found <a href="https://huggingface.co/datasets/dhmeltzer/yake_top3_asks_cleaned">here</a>.
  
You can also test the models yourselves using a Streamlit application hosted on Huggingface, click href="https://huggingface.co/spaces/dhmeltzer/qg_generation">here</a>. The application let's you compare the performance of two of our fine-tuned models, BART-Large and FLAN-T5-Base trained on the un-augmented dataset, with the zero-shot performance of FLAN-T5-XXL and GPT-3.5.

# Reports

Information on the complete workflow for this project are given in the following Weights and Biases <a href="https://api.wandb.ai/links/dmeltzer/7an677es">report</a>. There we detail the main ideas of the CONSISTENT paper, how to clean the r/AskScience dataset before training question generating models, and finally present several measures of how well our fine-tuned models perform on the validation split of the r/AskScience dataset.
We also test how well our models compare against the zero-shot performance of GPT-3.5 and FLAN-T5-XXL.
Overall, we find that our best performing model, BART-Large, produces coherent and interesting questions, but is still out-performed by GPT-3.5 and FLAN-T5-XXL. 

We also wrote the following Weights and Biases <a href="https://wandb.ai/dmeltzer/Question_Generation/reports/Exploratory-Data-Analysis-for-r-AskScience--Vmlldzo0MjQwODg1">report</a>, where we perform exploratory data analysis (EDA) on the cleaned r/AskScience dataset. For the most part the results we obtain are not used in the question generating process, but we think many of the results are interesting on their own and may also be useful in future work.
In particular, we performed topic modeling on the r/AskScience dataset using <a href="https://github.com/MaartenGr/BERTopic">BERTopic</a> and visualized the results using dimensional reduction, a heatmap, and heirarchial clustering. We also looked at what are the most common URLs cited in the posts and comments of r/AskScience, the score distribution of the top-rated comments, and finally the length distribution of titles, posts, and comments in the r/AskScience dataset.

# Structure of Repository

This repository is currently a work in progress and is still being updated. The <b>notebooks</b> folder contains the Google colab notebooks used to actually generate the main results. The file QG_consistent.ipynb contains the code used to download, process, and clean the r/AskScience dataset as well as the code used to actually train the model. The file EDA.ipynb contains results for exploratory data analysis on this cleaned dataset. Finally, the file inference.ipynb contains the code used to perform inference both with our fine-tuned models as well as with GPT-3.5 and FLAN-T5-XXL. The latter two models are too large to train and run inference on locally, so we call these models using the OpenAI and Huggingface APIs, respectively, and test them using zero-shot learning. These notebooks will be updated in the coming days to contain more comments and markdown text so they are more user-friendly.

The <b>scripts</b> folder contains essentially the same code as the QG_consistent.ipynb file, except the code has been refactored and commented to make it more readable. In particular, the python files in the <b>scripts</b> folder can be run on the command line. The file download_data.py contains the code to download the dataset from Huggingface, the file process_data contains the code to process and clean the dataset, and the file train.py contains the code to train the tranformer models.
