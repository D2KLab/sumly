# sumly


# Transformer-Model-Based-Extractive-Summarization-for-Clinical-Notes


                  
### Baseline approach Frequency based developed a CUI-based on extractive summarization of discharge summaries.


   ## Requirements
You can find requirement versions inside **requirements.txt** file.

> $ pip install -r requirements.txt

## Installing library

To be sure installed packages you can go and check it by clicking following steps. **PyCharm-->Preferences-->Project interpreter**. in case you need to install new packages you can add by clicking **"+"** there.
Some packet you should install via pip on terminal. 
> pip install name

If you face any issue while installing  **"en_core_web_lg"** use following commands in the terminal:

>python -m venv .env

>source .env/bin/activate

>pip install -U spacy

>python -m spacy download en_core_web_lg

In case you already created virtual environment in Pycharm start directly from here:

>pip install -U spacy

>python -m spacy download en_core_web_lg

## The steps to launch the application
### Get the code

You need to clone the repository:

> $ git clone https://github.com/D2KLab/sumly.git

Run **fsummary.py** on CLI. You can use PyCharm terminal or on already created environment.

Example: 
>$python fsummary.py file.txt file1.txt

in order to upload to doccano you can use **.jsonl** format instead of **.txt**

Example: 
>$python fsummary.py file.txt file1.jsonl

FYI your input file (file.txt) should be inside the same directory in order to execute successfully.

# Colaboratory Notebook

Since our local machine do not support GPU that's why we used CPU option and for that reason you should run **fsummary.py** on PyCharm or similar editor. 
However you can find  other baseline approach models on repository in order to run directly on Colab. 


# Transformed Based Bert Model

You can find Transformed based solution in **Main.ipynb**. You can run directly on Colaboratory. Since Colab supports GPU we run our model on Colab.
if you need help with Colab read this short article: https://medium.com/@rizvansaatov94/how-to-import-data-to-google-colab-for-the-beginner-6a311f051279

# BertViz

BertViz is a tool for visualizing Attention in the Transformer model, supporting all models from the transformers library (BERT, GPT-2, XLNet, RoBERTa, XLM, CTRL, etc.).

# Attention-head view
The Attention-head view visualizes the attention patterns produced by one or more attention heads in a given transformer layer.

![alt text](https://github.com/D2KLab/sumly/blob/main/Images/head_thumbnail_left.png) 
![alt text](https://github.com/D2KLab/sumly/blob/main/images/head_thumbnail_right.gif) 

The Attention view supports all models from the Transformers library.

# Model view

The model view provides a birds-eye view of Attention across all of the model’s layers and heads.

![alt text](https://github.com/D2KLab/sumly/blob/main/images/model_thumbnail.jpg) 

The model view supports all models from the Transformers library

# Neuron view

The neuron view visualizes the individual neurons in the query and key vectors and shows how they are used to compute Attention.

![alt text](https://github.com/D2KLab/sumly/blob/main/images/neuron_thumbnail.png)


# Conlcusion


We used clinical notes written by doctors as an input. You can find  **mimic-iii-clinical-database-demo** in reporsitory. This was general view of statistical-based and trasformed-based models. Our purpose is to boost the accuracy of our output. 

# References

- https://github.com/jessevig/bertviz
