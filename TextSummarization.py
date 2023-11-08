"""
In his 1958 paper "The automatic creation of literature abstracts" H.P. Luhn hypothesized hypothesized that the more
often words appear, the more crucial they are to the text's meaning. This is what is done in the following code:
it extracts the frequencies of each word in the sentences and retains the sentences with the highest frequency scores.
This is a simple, working technique, but it is not state-of-the-art. Other advanced text summarizers would use
word embedding to extract semantic significance.
For example, see the paper "Text Document Summarization Using Word Embedding" by Mudassir Mohd et al.
"""

import nltk  # Natural Language Toolkit is a popular Python library for NLP tasks
from nltk.tokenize import word_tokenize, sent_tokenize  # Modules used to make word and sentences tokenization
from nltk.corpus import stopwords  # Provide a data set of pre-determined stop words such as “the”, “a”, “an”, “in”
from collections import defaultdict  # Same than a classic dictionnary except for the fact that defaultdict never raises a KeyError.
from heapq import nlargest

# The input text is from the paper "Text document summarization using word embedding" by Mudasi Mohd and al.
input_text = "Another novel feature of our text summarizer is that it also removes redundancy by checking if the two sentences are written in the rephrased manner, it will eliminate those sentences. Since each cluster is composed of semantically similar sentences and if the ranking score of two sentences is almost similar then it implies both are conveying similar meaning and are thus included only once. This is an exciting and essential feature that will eliminate the semantically similar sentences by including them only once. It is an important feature especially producing summaries of long textual documents wherein authors tend to repeat the sentences by writing them in different ways, and their ranking score will be high, but our proposed system can identify them and discard them from the summaries despite their high ranks. We use sentence position as a metric to chose a sentence among the two semantically similar sentences. Whichever sentence has higher sentence position scores will be selected in summary."
aimed_num_sentences = 3  # Number of sentences in the summary

# Text Preprocessing
sentences = sent_tokenize(input_text)  # Tokenize the text into sentences

# Tokenize the text into words and filter out stopwords and punctuation
stop_words = set(stopwords.words('english'))
words = word_tokenize(input_text.lower())
words = [word for word in words if word.isalnum() and word not in stop_words]

# Count the word frequencies (excluding stopwords)
frequency = defaultdict(int)
for word in words:
    frequency[word] += 1

# Score sentences based on word frequency
sentence_scores = defaultdict(int)
for sentence in sentences:
    for word in word_tokenize(sentence.lower()):
        if word in frequency:
            sentence_scores[sentence] += frequency[word]

# Get the most important sentences
most_important_sentences = nlargest(aimed_num_sentences, sentence_scores, key=sentence_scores.get)

# Return the summary
print(' '.join(most_important_sentences))
