# sumly


# Statistical-Based and Transformer-Based Extractive Summarization of Clinical Notes
              

### Requirements 
The required packages and their versions may be found inside the **requirements.txt** file.

> $ pip install -r requirements.txt


### Installation

To check on the installed packages, the following steps may be followed: **PyCharm-->Preferences-->Project interpreter**. In case new packages need to be installed, they can be added by clicking the **"+"** button.

Some packages should be installed via pip using the terminal.

> $ pip install name

If there are issues while installing  **"en_core_web_md"**, use the following commands in the terminal:

> $ python -m venv .env

> $ source .env/bin/activate

> $ pip install -U spacy

> $ python -m spacy download en_core_web_md

In case a virtual environment has already been created in Pycharm, use the following commands instead:

> $ pip install -U spacy

> $ python -m spacy download en_core_web_md


### Usage

Clone the repository using:

> $ git clone https://github.com/D2KLab/sumly.git

Run **main.py** using the CLI. You can use the PyCharm terminal or an already created environment. The programm has two main commands:

For **article retrieval**:

> $ python main.py retrieve --query --max_articles --start_date --end_date --output_path --email

Example:

> $ python main.py retrieve --query 'chronic airways disease' --max_articles 10 --start_date 2010/01/01 --end_date 2020/01/01 --output_path articles --email 'abc@gmail.com'

An email must be specified as PubMed's information retrieval API requires one to prevent usage abuse.

For **text summarization**:

> $ python main.py summarize --input_path --output_path --method

The text files may be processed **individually** or **by batches** by chaging the input_path to be either a **single filename** or a **file directory** containing the text files.

> $ python main.py summarize --input_path **article_20169073.txt** --output_path summaries --method statistical

> $ python main.py summarize --input_path **articles** --output_path summaries --method statistical

The summarization method may also be chosen to be **statistical-based** or **transformer-based**.

> $ python main.py summarize --input_path **article_20169073.txt** --output_path summaries --method **statistical**

> $ python main.py summarize --input_path **article_20169073.txt** --output_path summaries --method **transformer**

To use the transforer-based method, the computer must have a **GPU**. Additionally, the input files should be in a text (**.txt**) format. 


### Colaboratory Notebook

The **Notebooks** folder containts all the notebooks for both the article retrieval and text summarization components of the program. In both subfolders, a **main.ipynb** may be found that can be run directly in Colaboratory. Other baseline approaches for the text summarization component may also be founder are run directly on Colaboratory.

If you wish to import data into Colaboratory, you may follow this [short guide](https://medium.com/@rizvansaatov94/how-to-import-data-to-google-colab-for-the-beginner-6a311f051279). If you wish to run your colab and jupyter file via the CLI, you may use the [**colab-cli**](https://github.com/Akshay090/colab-cli) library.


### Results

![alt text](https://github.com/D2KLab/sumly/blob/main/Images/cos_and_jacc.png) ![alt text](https://github.com/D2KLab/sumly/blob/main/Images/kld_and_jsd.png)

Left: Cosine Similarity and Jaccard Similarity where a higher value signifies a higher similarity. <br/>
Right: KLD and JSD divergence where a value closer to 0 signifies a higher similarity.


### BertViz

BertViz is a tool for visualizing Attention in a Transformer model and supports all models from the transformers library (BERT, GPT-2, XLNet, RoBERTa, XLM, CTRL, etc.).

#### Attention-head View

The Attention-head view visualizes the attention patterns produced by one or more attention heads in a given transformer layer.

![alt text](https://github.com/D2KLab/sumly/blob/master/Images/head_thumbnail_left.png) 
![alt text](https://github.com/D2KLab/sumly/blob/master/Images/head_thumbnail_right.gif) 

The Attention view supports all models from the Transformers library.

#### Model View

The model view provides a birds-eye view of Attention across all of the model’s layers and heads.

![alt text](https://github.com/D2KLab/sumly/blob/master/Images/model_thumbnail.jpg) 

The model view supports all models from the Transformers library.

#### Neuron View

The neuron view visualizes the individual neurons in the query and key vectors and shows how they are used to compute the Attention.

![alt text](https://github.com/D2KLab/sumly/blob/master/Images/neuron_thumbnail.png)


### Conclusion

We used clinical notes written by physicians as input. For a detailed description of the database structure, see the [MIMIC-III Clinical Database Page](https://physionet.org/content/mimiciii-demo/1.4/). These were the general view of statistical-based and transformed-based models. Our purpose was to summarize clinical notes to make it easier for readers, especially for physicians.


### References

- https://github.com/jessevig/bertviz
- https://github.com/Akshay090/colab-cli
- https://towardsdatascience.com/heres-how-i-made-a-cli-tool-to-work-with-google-colab-notebooks-7678a88ca662
