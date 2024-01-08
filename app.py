from flask import Flask, render_template, request
import math
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


app = Flask(__name__)


"""# Tokenize sentences and create a word frequency matrix for each sentence"""

def create_word_frequency_matrix(sentences):
    word_frequency_matrix = {}
    stop_words = set(stopwords.words("nepali"))

    for sent in sentences:
        word_freq_table = {}
        words = sent.split()
        for word in words:
            if word in stop_words:
                continue

            if word in word_freq_table:
                word_freq_table[word] += 1
            else:
                word_freq_table[word] = 1

        word_frequency_matrix[sent[:10]] = word_freq_table

    return word_frequency_matrix

"""# Create a term frequency matrix from the word frequency matrix"""

def create_term_frequency_matrix(word_frequency_matrix):
    term_frequency_matrix = {}

    for sent, word_freq_table in word_frequency_matrix.items():
        term_freq_table = {}

        count_words_in_sentence = len(word_freq_table)
        for word, count in word_freq_table.items():
            term_freq_table[word] = count / count_words_in_sentence

        term_frequency_matrix[sent] = term_freq_table

    return term_frequency_matrix

"""# Create a table of how many documents contain each word"""

def create_document_frequency_table(word_frequency_matrix):
    doc_frequency_table = {}

    for sent, word_freq_table in word_frequency_matrix.items():
        for word, count in word_freq_table.items():
            if word in doc_frequency_table:
                doc_frequency_table[word] += 1
            else:
                doc_frequency_table[word] = 1

    return doc_frequency_table

"""# Create an inverse document frequency matrix from the word frequency matrix and document frequency table"""

def create_inverse_document_frequency_matrix(word_frequency_matrix, doc_frequency_table, total_documents):
    inverse_document_frequency_matrix = {}

    for sent, word_freq_table in word_frequency_matrix.items():
        inverse_doc_freq_table = {}

        for word in word_freq_table.keys():
            inverse_doc_freq_table[word] = math.log10(total_documents / float(doc_frequency_table[word]))

        inverse_document_frequency_matrix[sent] = inverse_doc_freq_table

    return inverse_document_frequency_matrix

"""# Create a term frequency-inverse document frequency matrix from the term frequency matrix and inverse document frequency matrix"""

def create_term_frequency_inverse_document_frequency_matrix(term_frequency_matrix, inverse_document_frequency_matrix):
    tf_idf_matrix = {}

    for (sent1, term_freq_table1), (sent2, inverse_doc_freq_table2) in zip(term_frequency_matrix.items(), inverse_document_frequency_matrix.items()):
        tf_idf_table = {}

        for (word1, value1), (word2, value2) in zip(term_freq_table1.items(), inverse_doc_freq_table2.items()):
            tf_idf_table[word1] = float(value1 * value2)

        tf_idf_matrix[sent1] = tf_idf_table

    return tf_idf_matrix

"""# Calculate sentence scores based on TF-IDF"""

def calculate_sentence_scores(tf_idf_matrix):
    sentence_scores = {}

    for sent, tf_idf_table in tf_idf_matrix.items():
        total_score_per_sentence = 0

        count_words_in_sentence = len(tf_idf_table)
        for word, score in tf_idf_table.items():
            total_score_per_sentence += score
        if count_words_in_sentence != 0:
            sentence_scores[sent] = total_score_per_sentence / count_words_in_sentence
        else:
            sentence_scores[sent] = 0
    return sentence_scores

"""# Calculate the average score of sentences"""

def calculate_average_sentence_score(sentence_scores):
    """Calculate the average score of sentences."""
    sum_values = sum(sentence_scores.values())
    average_score = (sum_values / len(sentence_scores))
    return average_score

"""
# Generate a summary based on the sentence scores and a given threshold"""

def generate_summary(sentences, sentence_scores, threshold):
    """Generate the summary based on the sentence scores."""
    summary = []

    for sentence in sentences:
        if sentence[:10] in sentence_scores and sentence_scores[sentence[:10]] >= (threshold):
            summary.append(sentence)

    return '।'.join(summary)
    
    
def summarize_text(text):
    sentences = re.split('।', text)
    word_frequency_matrix = create_word_frequency_matrix(sentences)
    term_frequency_matrix = create_term_frequency_matrix(word_frequency_matrix)
    doc_frequency_table = create_document_frequency_table(word_frequency_matrix)
    inverse_document_frequency_matrix = create_inverse_document_frequency_matrix(word_frequency_matrix, doc_frequency_table, len(sentences))
    tf_idf_matrix = create_term_frequency_inverse_document_frequency_matrix(term_frequency_matrix, inverse_document_frequency_matrix)
    sentence_scores = calculate_sentence_scores(tf_idf_matrix)
    threshold = calculate_average_sentence_score(sentence_scores)
    summary = generate_summary(sentences, sentence_scores, 0.8 * threshold)
    return summary

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    if request.method == 'POST':
        text = request.form['text']
        summary = summarize_text(text)
        return summary

if __name__ == '__main__':
    app.run(debug=True)
