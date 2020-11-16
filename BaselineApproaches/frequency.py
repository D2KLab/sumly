#!/usr/bin/python

#tocompare
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
text_str = '''

90 15 15472 166833.0 2176-08-04   Discharge summary Report   Admission Date:  [**2176-7-30**]     Discharge Date:  [**2176-8-4**]

Date of Birth:   [**2114-2-8**]     Sex:  M

Service:  [**Hospital1 212**]

HISTORY OF PRESENT ILLNESS:  The patient is a 62 year-old
male with a past medical history of squamous cell lung cancer
treated with right total pneumonectomy  chronic obstructive
pulmonary disease on 2 to 3 liters of home oxygen with
saturations in the low 90s at baseline  congestive heart
failure  and diabetes mellitus type 2 who was recently
admitted from [**7-15**] to [**2176-7-19**] for presumed
bronchitis or bronchiectasis flare here with recurrent cough 
shortness of breath and fevers.  During his last admission
two weeks ago he was treated for chronic obstructive
pulmonary disease flare versus bronchitis with a ten day
Prednisone taper and Augmentin for one week.  He underwent
bronchoscopy due to concern for possible endobronchial
lesion  which was normal.  Sputum sample was done at that
time showed no growth.  He was discharged at his baseline
function on [**2176-7-19**].  The plan was to treat him for one
week of Augmentin  skip one week followed by Bactrim for one
week  skip one week and then on Augmentin for two weeks for
pneumonia prophylaxis.  The last dose of Augmentin was
[**2176-7-22**] after being on Augmentin for only three days.

He was doing well until approximately one week ago when he
developed mild spasms in the afternoon that he thought was
due to low potassium.  Within the following days he
complained of worsening cough productive of clear sputum.  He
had a low grade temperature  mild headache and worsening
cough and presented to the Emergency Department.  He denied
any sinus pain  sore throat  chest pain  abdominal pain 
diarrhea  dysuria or joint pain.  In the Emergency Department
he was febrile to 102 orally and had a heart rate of 160 and
a blood pressure of 118/56.  Respiratory rate 28.  Sating 88
to 98% on 100% nonrebreather.  Initially he was stable  but
then had a gradual change in mental status with hypoxia 
which resulted in his elective intubation.  He received Lasix
100 mg intravenous twice  1 mg of Bumex and 1 gram of
Ceftriaxone as well as 125 mg of Solu-Medrol.  He was also
placed on a heparin drip for a subtherapeutic INR and given
morphine and Ativan for sedation.  Chest x-ray showed no
focal pneumonia or evidence of heart failure.  The patient
then underwent a CT angiogram of the chest that showed no
evidence of pulmonary embolism.
'''

def _create_frequency_table(text_string) -> dict:
    """
    we create a dictionary for the word frequency table.
    For this, we should only use the words that are not part of the stopWords array.
    Removing stop words and making frequency table
    Stemmer - an algorithm to bring words to its root word.
    :rtype: dict
    """
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text_string)
    ps = PorterStemmer()

    freqTable = dict()
    for word in words:
        word = ps.stem(word)
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

    return freqTable
def _score_sentences(sentences, freqTable) -> dict:
    """
    score a sentence by its words
    Basic algorithm: adding the frequency of every non-stop word in a sentence divided by total no of words in a sentence.
    :rtype: dict
    """

    sentenceValue = dict()

    for sentence in sentences:
        word_count_in_sentence = (len(word_tokenize(sentence)))
        word_count_in_sentence_except_stop_words = 0
        for wordValue in freqTable:
            if wordValue in sentence.lower():
                word_count_in_sentence_except_stop_words += 1
                if sentence[:10] in sentenceValue:
                    sentenceValue[sentence[:10]] += freqTable[wordValue]
                else:
                    sentenceValue[sentence[:10]] = freqTable[wordValue]

        if sentence[:10] in sentenceValue:
            sentenceValue[sentence[:10]] = sentenceValue[sentence[:10]] / word_count_in_sentence_except_stop_words

        '''
        Notice that a potential issue with our score algorithm is that long sentences will have an advantage over short sentences. 
        To solve this, we're dividing every sentence score by the number of words in the sentence.
        
        Note that here sentence[:10] is the first 10 character of any sentence, this is to save memory while saving keys of
        the dictionary.
        '''

    return sentenceValue
def _find_average_score(sentenceValue) -> int:
    """
    Find the average score from the sentence value dictionary
    :rtype: int
    """
    sumValues = 0
    for entry in sentenceValue:
        sumValues += sentenceValue[entry]

    # Average value of a sentence from original text
    average = (sumValues / len(sentenceValue))

    return average


def _generate_summary(sentences, sentenceValue, threshold):
    sentence_count = 0
    summary = ''

    for sentence in sentences:
        if sentence[:10] in sentenceValue and sentenceValue[sentence[:10]] >= (threshold):
            summary += " " + sentence
            sentence_count += 1

    return summary
def run_summarization(text):
    # 1 Create the word frequency table
    freq_table = _create_frequency_table(text)

    '''
    We already have a sentence tokenizer, so we just need 
    to run the sent_tokenize() method to create the array of sentences.
    '''

    # 2 Tokenize the sentences
    sentences = sent_tokenize(text)

    # 3 Important Algorithm: score the sentences
    sentence_scores = _score_sentences(sentences, freq_table)

    # 4 Find the threshold
    threshold = _find_average_score(sentence_scores)

    # 5 Important Algorithm: Generate the summary
    summary = _generate_summary(sentences, sentence_scores, 0.8* threshold)

    return summary

if __name__ == '__main__':
    result = run_summarization(text_str)
    print(result)

