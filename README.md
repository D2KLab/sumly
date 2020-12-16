# sumly


# Transformer-Based Model and Statistical-Based Model Extractive Summarization for Clinical Notes
              

 #### Requirements 
You can find requirement versions inside **requirements.txt** file.

> $ pip install -r requirements.txt

#### Installation

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


#### Usage

You need to clone the repository:

> $ git clone https://github.com/D2KLab/sumly.git

Run **main.py** on CLI. You can use PyCharm terminal or on already created environment.
By choosing you can pilot either the statistical-based model or the transformer-based model.

Example: 
>$python main.py file.txt file1.txt **statistical**

>$python main.py file.txt file1.txt **transformer**

In order to upload to doccano you can choose **.jsonl** format instead of **.txt**

Example: 
>$python main.py file.txt file1.jsonl **statistical/transformer**

**FYI** your input file (file.txt) should be inside the same directory in order to execute successfully.
Just be sure your computer supports GPU otherwise this option does **NOT** work. In addition input file should be in text (**.txt**) format. 

#### Colaboratory Notebook

In **Notebooks** folder located all notebooks file. You can find Transformed based solution in **Main.ipynb**. You can run directly on Colaboratory. Since Colab supports GPU we run our model on Colab.
if you need help with Colab read this short article: [Medium](https://medium.com/@rizvansaatov94/how-to-import-data-to-google-colab-for-the-beginner-6a311f051279).
Furthermore, you can find  other baseline approach models on repository in order to run directly on Colab. 
If you want to run your colab and jupyter file via CLI you can use **Colab-cli** library.
See repository here: [colab-cli](https://github.com/Akshay090/colab-cli)

#### Result

![alt text](https://github.com/D2KLab/sumly/blob/main/Images/download.png)

#### BertViz

BertViz is a tool for visualizing Attention in the Transformer model, supporting all models from the transformers library (BERT, GPT-2, XLNet, RoBERTa, XLM, CTRL, etc.).

#### Attention-head view
The Attention-head view visualizes the attention patterns produced by one or more attention heads in a given transformer layer.

 
![alt text](https://github.com/D2KLab/sumly/blob/master/Images/head_thumbnail_left.png) 
![alt text](https://github.com/D2KLab/sumly/blob/master/Images/head_thumbnail_right.gif) 


The Attention view supports all models from the Transformers library.

#### Model view

The model view provides a birds-eye view of Attention across all of the model’s layers and heads.


![alt text](https://github.com/D2KLab/sumly/blob/master/Images/model_thumbnail.jpg) 


The model view supports all models from the Transformers library

#### Neuron view

The neuron view visualizes the individual neurons in the query and key vectors and shows how they are used to compute Attention.


![alt text](https://github.com/D2KLab/sumly/blob/master/Images/neuron_thumbnail.png)



#### Conlcusion


We used clinical notes written by doctors as an input. For a detailed description of the database structure, see the [ MIMIC-III Clinical Database Page](https://physionet.org/content/mimiciii-demo/1.4/). These were general view of statistical-based and transformed-based models. Our purpose was to summarize of clinical notes in order to make it easier for readers, especially for doctors. :+1: 

#### References

- https://github.com/jessevig/bertviz
- https://github.com/Akshay090/colab-cli
- https://towardsdatascience.com/heres-how-i-made-a-cli-tool-to-work-with-google-colab-notebooks-7678a88ca662
