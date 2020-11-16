# sumly
# Transformer-Model-Based-Extractive-Summarization-for-Clinical-Notes


                  
### Baseline approach Frequency based summarization for running via the CLI and for comparing with Transformed based approach



   ## Requirements
You can find requirement versions inside requirements.txt file.
> $ pip install -r requirements.txt

## Installing library

To be sure installed packages you can go and check it by clicking following steps. PyCharm-->Preferences-->Project interpreter. in case you need to install new packages you can add by clicking "+" there.
Some packet you should install via pip on terminal. 
> pip install name

If you face any issue while installing  "en_core_web_lg" use following commands in the terminal:

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

Run fsummary.py on CLI. You can use PyCharm terminal or on already created environment.

Example: 
>$python fsummary.py file.txt file1.txt

FYI your input file (file.txt) should be inside the same directory in order to execute successfully.

# Colaboratory Notebook

Since our local machine do not support GPU that's why we used CPU option and for that reason you should run **fsummary.py** on PyCharm or similar editor. 
However you can find  other tested models on repository in order to run directly on Colab. 

# Conlcusion

Our purpose is to boost the accuracy of our output. We used clinical notes written by doctors as an input. You can find  mimic-iii-clinical-database-demo in reporsitory. We have used different models and metrics that already devoloped or we modified them. Now we are seeking for new way in order to get better result. Research going on.
