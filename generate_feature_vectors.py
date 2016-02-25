from nltk.tokenize import word_tokenize, RegexpTokenizer
from bs4 import BeautifulSoup
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
import time
import re 
import string

'''
    This script generates numeric feature matricies using the Reuters articles.
    
    The end result contains two numeric matricies ( feature_matrix_count_large, feature_matrix_count_small) that correspond to the word count for each document. Ie, feature_matrix_count_small[i] contains the word counts of the 200 most common words (of the corpus) for the ith document in the dataset.
    
    The size of each matrix is (number of articles-by-number of featured words).
    
'''

t0 = time.time()

corpus_tokens = list()
response_vector= list();

# Majority of files are commented out to reduce runtime
fileNumbers=['000','001'] #,'002','003','004','005','006','007','008','009','010','011','012','013','014','015','016','018','019','020','021']
tokenizer = RegexpTokenizer(r'\w+')
body_tokens = []
feature_matrix_tokens = list()
for index in fileNumbers:
    file_path='sgm_files/reut2-'+index+'.sgm'
    soup=BeautifulSoup(open(file_path), "html.parser")
    articles = soup.findAll('reuters')

    for curarticle in articles:
        scur = str(curarticle)
        if ('</dateline>' in scur):
            body_tokens = list()
            start_index = scur.index('</dateline>')
            body_text = scur[start_index+11:len(scur)-25]
            body_tokens = tokenizer.tokenize(body_text)
            
            #create entry in feature matrix for current article
            title = curarticle.title
            t = curarticle.topics 
            for i in t:
                to_replace = i.get_text()
                body_text = body_text.replace(to_replace, '')
            
            # process article text to remove unneeded characters
            body_text.lower()
            body_text = body_text.replace('    ',' ') .replace('\n',' ').replace('\t',' ')
            body_text =''.join([i for i in body_text if not i.isdigit()])
            body_text = re.sub('[%s]' % re.escape(string.punctuation), '', body_text)
            
            # add current tokens to the total corpus token list 
            body_tokens = tokenizer.tokenize(body_text)
            corpus_tokens = corpus_tokens + body_tokens #add current article's tokens to total corpus
            feature_matrix_tokens.append(body_tokens)
            
            #create response vector for current article
            t_text = list()
            for i in t:
                t_text.append(i.text)
            if len(t_text) == 0:
                response_vector.append(title.text.split(' ', 1)[0])
            else:
                response_vector.append(" ".join(i for i in t_text))

porter_stemmer = PorterStemmer()
fdist = FreqDist(corpus_tokens); #get distribution of entire corpus

print("Initial word count: {}".format(len(fdist)))
filtered_words1 = [word.lower() for word in fdist if word.lower() not in stopwords.words('english')] #eliminate stop words from corpus tokens
print("After eliminating stop words: {}".format(len(filtered_words1)))
filtered_words2 = []
for word in filtered_words1:
    filtered_words2.append(porter_stemmer.stem(word))
fdist2 = FreqDist(filtered_words2)
print("After using Porter stemming: {}".format(len(fdist2)))
important_words_large = [word for word in fdist2 if fdist2[word]> 4 and len(word) > 3]

print("Only retain words that appear more than four times in the corpus and are longer than 3 characters: {}".format(len(important_words_large)))

important_words_small = fdist2.most_common(200)

# Create feature matricies for word count of each document
feature_matrix_count_small = list()
feature_matrix_count_large = list()  
for line in feature_matrix_tokens: #iterate through all articles in matrix
    curline_large=list()
    curline_small=list()
    for word in important_words_large:
        curline_large.append(line.count(word))
    for word in important_words_small:
        curline_small.append(line.count(word))
    feature_matrix_count_large.append(curline_large)
    feature_matrix_count_small.append(curline_small)
t1 = time.time()

print("Time of iteration (s): {}".format(t1-t0))
