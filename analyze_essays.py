"""
uses text analytics on final essays of Fall 2018 RCC 200 course final research papers

to import textract:
pip3 install pdfminer3k
sudo apt-get install libpulse-dev swig
clone github repo (https://github.com/deanmalmgren/textract)
python3 setup.py install

could also use textstat package for readability calculations
readability:
Flesch-Kincaid formula --
words per sentence
average syllables per word
FKRA = (0.39 x ASL) + (11.8 x ASW) - 15.59

Zipfian distribution

bi- and trigrams from PMI (looking for repetative phrases)

Topic modeling (LDA)

TODO (probably will never do it, but ideas are there):

average word length (letters)

similarity (TFIDF and cosine distance)

overall sentiment, and distribution of sentiment of sentences

wordcloud of word lemmas

progressive tense used

passive sentences

bar plot of first word of sentences
"""
import re
import glob
import string

import numpy as np
# for syllables count
import pyphen
dic = pyphen.Pyphen(lang='en')
import spacy
nlp = spacy.load('en_core_web_lg')
import textract
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import matplotlib.pyplot as plt
from scipy.stats import zipf
import nltk.collocations as nc



def load_papers(directory='/home/nate/Dropbox/regis/RCC200/writings_by_student/', subpath='project2', text_only=True):
    """
    Loads .doc, .docx, and .pdf papers into dictionary with student's name as key and
    paper string as value.  Sample papers in ./papers/ directory from
    https://www.mesacc.edu/~paoih30491/ArgumentSampleEssays.html

    Keyword arguments:
    directory -- string; path to papers -- should be folders with studets' names like Last, First
    subpath -- if files are in a folder within each student folder, this should be the path to that folder
    text_only -- Boolean, if True, will load text.  Otherwise will use textract
                and python-docx
    """
    folders = glob.glob(directory + '*')
    papers = {}
    for f in folders:
        student = f.split('/')[-1].split(', ')[1]
        print(student)
        path = f
        if subpath:
            path += '/{}/*'.format(subpath)

        if text_only:
            path += '.txt'

        files = glob.glob(path)
        if len(files) > 1:
            print('uh-oh, more than one paper; using first one')

        if text_only:
            with open(files[0], 'rb') as text:
                papers[student] = text.read().decode('utf-8')
        else:
            papers[student] = load_word_doc_pdf(files[0])

    return papers


def load_word_doc_pdf(path):
    """
    original way to load data; but realized the headers and references are all
    different.  Manually extracted raw text into .txt files instead
    """
    try:
        paper = textract.process(path)
    except:  # one paper with docx errors...
        doc = Document(path)
        text = ''
        for p in doc.paragraphs:
            text += p.text

    return text


def clean_text(papers):
    # dict with lemmatized words and spaces removed
    lemma_papers = {}
    # spacy-processed papers
    spacy_papers = {}
    raw_spacy_papers = {}
    # for removing punctuation and numbers
    table = str.maketrans({key: None for key in string.punctuation + string.digits})
    for k in papers.keys():
        # replace any whitespace characters with just one space
        paper = re.sub('[\s]+', ' ', papers[k])
        raw_spacy_papers[k] = nlp(paper)
        lowercased_doc = paper.lower()
        # removes punctuation and numbers
        clean_document = lowercased_doc.translate(table)
        spacy_papers[k] = nlp(clean_document)

        lemma_papers[k] = [w.lemma_ if w.lemma_ != '-PRON-' else w.lower_ for w in spacy_papers[k]]

    return raw_spacy_papers, spacy_papers, lemma_papers


def get_Flesch_Kincaid(spacy_papers):
    """
    gets Flesch-Kincaid readability score for dictionary of spacy papers

    FKRA = (0.39 x ASL) + (11.8 x ASW) - 15.59
    """
    sent_lengths = {}
    syllables = {}
    readability = {}
    asl = {}
    asw = {}
    readability = {}
    for k in spacy_papers.keys():
        for sent in spacy_papers[k].sents:
            sent_lengths.setdefault(k, []).append(len(sent))
            syllables.setdefault(k, []).extend([len(dic.inserted(w.text).split('-')) for w in sent])

        asl[k] = np.mean(np.array(sent_lengths[k]).flatten())
        asw[k] = np.mean(np.array(syllables[k]).flatten())
        readability[k] = (0.39 * asl[k]) + (11.8 * asw[k]) - 15.59

    return sent_lengths, syllables, readability, asl, asw


def get_Flesch_Kincaid_one(paper):
    """
    gets Flesch-Kincaid readability score for one paper

    Keyword arguments:
    paper -- spacy-processed text
    """
    sent_lengths = []
    syllables = []
    for sent in paper.sents:
        sent_lengths.append(len(sent))
        syllables.extend([len(dic.inserted(w.text).split('-')) for w in sent])

    asl = np.mean(np.array(sent_lengths).flatten())
    asw = np.mean(np.array(syllables).flatten())
    readability = (0.39 * asl) + (11.8 * asw) - 15.59
    return asl, asw, readability


def make_zipf_plot(counts, tokens, title=None, savepath='./', save=False):
    """
    makes Zipfian distribution plot
    """
    # A Zipf plot
    # adapted from here: https://finnaarupnielsen.wordpress.com/2013/10/22/zipf-plot-for-word-counts-in-brown-corpus/
    # get counts for x and y
    ranks = np.arange(1, len(counts) + 1)
    indices = np.argsort(-counts)
    normalized_frequencies = counts[indices] / sum(counts)


    # make plot
    f = plt.figure(figsize=(10, 10))
    plt.loglog(ranks, normalized_frequencies, marker=".")

    # add the expected Zipfian distribution from the equation
    # 1.07 is usually a good bet for the shape parameter
    plt.loglog(ranks, [z for z in zipf.pmf(ranks, 1.07)])

    # add labels for clarity
    plt.xlabel("Frequency rank of token")
    plt.ylabel("Absolute frequency of token")

    ax = plt.gca()  # get current axis
    ax.set_aspect('equal')  # make the plot square
    plt.grid(True)
    if title is not None:
        plt.title(title)
    else:
        title = 'zipf_plot'  # for saving figure
        plt.title("Zipf plot")

    # add text labels
    last_freq = None
    for i in list(np.logspace(-0.5, np.log10(len(counts) - 1), 10).astype(int)):
        if last_freq != normalized_frequencies[i]:  # ensure words don't overlap...make sure y-val is different
            dummy = plt.text(ranks[i], normalized_frequencies[i], " " + tokens[indices[i]],
                             verticalalignment="bottom",
                             horizontalalignment="left")
        last_freq = normalized_frequencies[i]

    if save:
        plt.savefig(savepath + title + '.png')

    plt.show()


def get_top_grams(docs, n=2, top=20):
    """
    gets top "top" n-grams from the docs
    docs should be lemmatized ideally

    Keyword arguments:
    docs -- list or numpy array of strings (documents)
    n -- ngram length
    top -- number of top words to print out

    Returns:
    list of n-grams, list of counts (both sorted from greatest to least counts)
    """
    v = CountVectorizer(ngram_range=(n, n))
    grams = v.fit_transform(docs)
    # convert to array and flatten to avoid weird indexing
    gram_sum = np.array(np.sum(grams, axis=0)).flatten()
    gram_dict = {i: v for v, i in v.vocabulary_.items()}  # dictionary of index: word
    top_grams = gram_sum.argsort()[::-1]
    for i in top_grams[:top]:
        print('"' + gram_dict[i] + '" shows up', gram_sum[i], 'times')

    return [gram_dict[i] for i in top_grams], gram_sum[top_grams]


if __name__ == '__main__':
    papers = load_papers()
    raw_spacy_papers, spacy_papers, clean_papers = clean_text(papers)
    sent_lengths, syllables, readability, asl, asw = get_Flesch_Kincaid(raw_spacy_papers)
    words, counts = get_top_grams([spacy_papers['Aryan'].text], n=1, top=3)
    make_zipf_plot(counts, words, title='Zipf plot of Aryan\'s essay')
    for stu in spacy_papers.keys():
        words, counts = get_top_grams([spacy_papers[stu].text], n=1, top=3)
        ti = 'Zipf plot of {}\'s essay'.format(stu)
        path = '/home/nate/Dropbox/regis/RCC200/zipf/'
        make_zipf_plot(counts, words, title=ti, savepath=path, save=True)


    bigram_measures = nc.BigramAssocMeasures()
    trigram_measures = nc.TrigramAssocMeasures()
    finder = nc.BigramCollocationFinder.from_documents([[word.text for word in spacy_papers['Aryan']]])
    print('top 10 2-grams by PMI')
    top_bigrams = finder.nbest(bigram_measures.pmi, 10)
    if finder.ngram_fd.N(top_bigrams[0]) > 1:
        print('counts top bigram by PMI appears:')
        print(finder.ngram_fd.N(top_bigrams[0]))


    # analyze my essay for comparison
    with open('/home/nate/Dropbox/regis/RCC200/essays/short_assignment_1_turing/raw_text.txt', 'rb') as f:
        text = f.read().decode('utf-8')

    essays = {}
    essays['nate'] = text
    raw_spacy_papers, spacy_papers, clean_papers = clean_text(essays)
    sent_lengths, syllables, readability, asl, asw = get_Flesch_Kincaid(raw_spacy_papers)
    words, counts = get_top_grams([spacy_papers['nate'].text], n=1, top=3)
    make_zipf_plot(counts, words, title='Zipf plot of Dr. George\'s essay', savepath=path, save=True)
