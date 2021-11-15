import time
import urllib
import numpy as np
import pandas as pd
from nltk import tokenize
from bs4 import BeautifulSoup
from urllib.request import urlopen
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

stop_words = ['I', 'a', 'about', 'an', 'are', 'as', 'at', 'be', 'by', 'com', 'for', 'from', 'how', 'in', 'is', 'it',
              'of', 'on', 'or', 'that', 'the', 'this', 'to', 'was', 'what', 'when', 'where', 'who', 'will', 'with',
              'the', 'www']
N = 10


def get_text(url):
    time.sleep(5)
    seed_data = ''
    soup = BeautifulSoup(urllib.request.urlopen(url).read(), "html.parser")
    for tag in soup.findAll(('p', 'div')):
        if tag.name == 'div' and 'id' in tag.attrs:
            if 'toc' in tag['id']:
                break
        elif tag.name == 'p':
            seed_data += tag.getText().lower().strip()
    tokenized_words = tokenize.word_tokenize(seed_data)
    processed_data = []
    for word in tokenized_words:
        if word not in stop_words:
            x = PorterStemmer().stem(word)
            if x.isalnum():
                processed_data.append(x)
    return ' '.join(processed_data)


def get_reciprocal(seed_title, children):
    reciprocal = ["No"] * len(children)
    for i in range(len(children)):
        back_links = get_urls(children[i])[1]
        for j in back_links:
            if j == seed_title:
                reciprocal[i] = "Yes"
    return reciprocal


def get_urls(url):
    time.sleep(5)
    soup = BeautifulSoup(urllib.request.urlopen(url).read(), "html.parser")
    remove = ["/help:", "#cite", "/file:", "//upload", "/category"]
    children_page = []
    page_titles = []
    for p in soup.findAll('p'):
        for a in p.findAll('a'):
            if a.has_attr('href'):
                cite = a['href'].lower()
                if not any(x in cite for x in remove):
                    if len(children_page) < N:
                        title = a['href'][6:].replace('_', " ")
                        if title not in page_titles:
                            page_titles.append(title)
                            children_page.append("https://en.wikipedia.org" + a['href'])
                    else:
                        break
    return children_page, page_titles


def cos_sim(doc_matrix):
    query_tf = doc_matrix[0]
    A = np.delete(doc_matrix.T, 0, 1)
    lq = []
    for i in query_tf:
        if i == 0:
            lq.append(0)
        else:
            lq.append(np.log10(i) + 1)
    lq = np.matrix(lq).T
    ld = []
    for i in A.T:
        d = []
        for j in i:
            if j == 0:
                d.append(0)
            else:
                d.append(np.log10(j) + 1)
        ld.append(d)
    td = []
    for i in A:
        d = 0
        for j in i:
            if j > 0:
                d += 1
        if d == 0:
            td.append(0)
        else:
            td.append(np.log10(N / d))
    ltd = []
    for i in range(len(ld)):
        d = []
        for j in range(len(ld[0])):
            x = ld[i][j] * td[j]
            d.append(x)
        ltd.append(d)
    similarities = []
    query = np.matrix(lq).T
    norm_q = np.linalg.norm(query)
    for i in range(N):
        doc = np.matrix(ltd)[i].T
        numerator = query @ doc
        denominator = norm_q * np.linalg.norm(doc)
        similarities.append(float(numerator / denominator))
    return similarities


def main(seed, num):
    global N
    N = int(num)
    seed_title = seed[30:].replace("_", " ")
    children, page_titles = get_urls(seed)
    reciprocal = get_reciprocal(seed_title, children)
    docs = [get_text(seed)]
    for link in children:
        docs.append(get_text(link))
    v = CountVectorizer()
    x = v.fit_transform(docs)
    df = pd.DataFrame(x.toarray(), columns=v.get_feature_names())
    similarities = cos_sim(df.to_numpy())
    df = pd.DataFrame(data={'Child page': page_titles,
                            'Cosine similarity': similarities,
                            'Reciprocal links': reciprocal})
    print(df.to_string())


main(input("Wiki URL to crawl: "), input("Choose number of children pages (5-15 recommended): "))
