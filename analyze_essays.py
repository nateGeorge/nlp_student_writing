"""
uses text analytics on final essays of Fall 2018 RCC 200 course final research papers

to import textract:
pip3 install pdfminer3k
clone github repo (https://github.com/deanmalmgren/textract)
python3 setup.py install


readability:
Flesch-Kincaid formula --
words per sentence
average syllables per word
FKRA = (0.39 x ASL) + (11.8 x ASW) - 15.59


average word length (letters)

Zipfian distribution

Topic modeling (LDA)

similarity (TFIDF and cosine distance)

overall sentiment, and distribution of sentiment of sentences

wordcloud of word lemmas

progressive tense used

passive sentences

bar plot of first word of sentences

bi- and trigrams from PMI (looking for repetative phrases)
"""

import glob

# for syllables count
import pyphen
dic = pyphen.Pyphen(lang='en')
import spacy
nlp = spacy.load('en_core_web_lg')

# import textract
# from docx import Document


def load_papers(directory='/home/nate/Dropbox/regis/RCC200/writings_by_student/'):
    """
    Loads .doc, .docx, and .pdf papers into dictionary with student's name as key and
    paper string as value.  Sample papers in ./papers/ directory from
    https://www.mesacc.edu/~paoih30491/ArgumentSampleEssays.html
    """
    folders = glob.glob(directory + '*')
    papers = {}
    for f in folders:
        student = f.split('/')[-1].split(', ')[1]
        print(student)
        files = glob.glob(f + '/project2/*.txt')
        if len(files) > 1:
            print('uh-oh, more than one final paper; using first one')

        with open(files[0], 'rb') as text:
            papers[student] = text.read().decode('utf-8')

    return papers


def clean_text(papers):
    # dict with lemmatized words and spaces removed
    clean_papers = {}
    # spacy-processed papers
    spacy_papers = {}
    for k in papers.keys():
        spacy_papers[k] = nlp(papers[k])
        clean_papers[k] = [w.lemma_ if w.lemma_ != '-PRON-' else w.lower_ for w in spacy_papers[k]]

    return spacy_papers, clean_papers


def get_Flesch_Kincaid(spacy_papers):
    """
    gets Flesch-Kincaid readability score

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

    return sent_lengths, syllables, readability, asl, asw, readability


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


def load_word_docs_pdfs():
    """
    original way to load data; but realized the headers and references are all
    different.  Manually extracted raw text into .txt files instead
    """
    try:
        papers[student] = textract.process(paper)
    except:  # one paper with docx errors...
        doc = Document(paper)
        text = ''
        for p in doc.paragraphs:
            text += p.text

            papers[student] = text
