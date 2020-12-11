# Thanks to Professor Carolyn Sealfon of providing the foundation code of running with Naive Bayes classifier, vectorizing, K-means clustering, printing out example responses and clusters!
import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy
print('numpy: {}'.format(numpy.__version__))
# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas
print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))
import nltk

# load data
import csv

from google.colab import files
csvfile=files.upload()
csvfile2=open('rev02lab.csv', 'rt', newline='')
csvfile=files.upload()
csvfile3=open('rev03lab.csv', 'rt', newline='')
csvfile=files.upload()
csvfile4=open('rev04lab.csv', 'rt', newline='')
csvfile=files.upload()
csvfile5=open('rev05lab.csv', 'rt', newline='')
csvfile=files.upload()
csvfile6=open('rev06lab.csv', 'rt', newline='')
csvfile=files.upload()
csvfile7=open('rev07lab.csv', 'rt', newline='')
csvfile=files.upload()
csvfile8=open('rev08lab.csv', 'rt', newline='')
file = csv.reader(csvfile, delimiter=',')

# load data

# 1 = non-answer
# 2 = logistics (now combined with 3)
# 3 = content question
# https://nlp.stanford.edu/IR-book/html/htmledition/the-bernoulli-model-1.html
 
from sklearn.model_selection import train_test_split
 
ans1=[] #the student answers, each answer in a row
target=[]
for row in file:
    #print(row[0])
    ans1.append(str(row[0]))
    target.append(row[1])
csvfile.close() 

#add labeled data from other weeks

csvfile=open('rev03lab.csv', 'rt', newline='') #labeled data from pre-class 3
file3 = csv.reader(csvfile, delimiter=',')

#print(len(ans1))

for row in file3:
    #print(row[0])
    ans1.append(str(row[0]))
    target.append(row[1])
csvfile.close() 

#print(len(ans1))

csvfile=open('rev04lab.csv', 'rt', newline='') #labeled data from pre-class 4
file4 = csv.reader(csvfile, delimiter=',')

for row in file4:
    #print(row[0])
    ans1.append(str(row[0]))
    target.append(row[1])
csvfile.close()  

#print(len(ans1))

csvfile=open('rev05lab.csv', 'rt', newline='') #labeled data from pre-class 5
file5 = csv.reader(csvfile, delimiter=',')

for row in file5:
    #print(row[0])
    ans1.append(str(row[0]))
    target.append(row[1])
csvfile.close()      

#print(len(ans1))


csvfile=open('rev06lab.csv', 'rt', newline='') #labeled data from pre-class 6
file6 = csv.reader(csvfile, delimiter=',')

for row in file6:
    #print(row[0])
    ans1.append(str(row[0]))
    target.append(row[1])
csvfile.close()  

#print(len(ans1))

csvfile=open('rev07lab.csv', 'rt', newline='') #labeled data from pre-class 7
file7 = csv.reader(csvfile, delimiter=',')

for row in file7:
    #print(row[0])
    ans1.append(str(row[0]))
    target.append(row[1])
csvfile.close()  

#print(len(ans1))
csvfile=open('rev08lab.csv', 'rt', newline='') #labeled data from pre-class 8
file8 = csv.reader(csvfile, delimiter=',')

for row in file8:
    #print(row[0])
    ans1.append(str(row[0]))
    target.append(row[1])
csvfile.close()  


# build training, test dataset

from sklearn.model_selection import train_test_split
def train(classifier, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                        test_size=0.005, random_state=23)
 
    classifier.fit(X_train, y_train)
    print ("Accuracy: %s" % classifier.score(X_test, y_test))
    return classifier
 

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
trial1 = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', MultinomialNB(alpha=0.05)),
])

# trial for the classifier    
trial3 = Pipeline([
    ('vectorizer', TfidfVectorizer(ngram_range=(1,2))),
    ('classifier', BernoulliNB()),
])

# Naive Bayes classifier
BernNB = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', BernoulliNB(alpha=0.000001)),
])
from sklearn.linear_model import SGDClassifier

# New classifier I tried
# sgd = Pipeline([('vect', CountVectorizer()),
#                 ('tfidf', TfidfTransformer()),
#                 ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
#                ])
# sgd.fit(X)

# text pre-processing
from gensim.utils import simple_preprocess
#and stem the words
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
pro_sents=[simple_preprocess(row) for row in ans1]
sentsFilt=[] #will make into 2D array
for l in range(len(pro_sents)):
    sent=[] # 1D array of each sentence
    for w in range(len(pro_sents[l])):
        word = porter.stem(pro_sents[l][w])
        sent.append(word)
    sentsFilt.append(str(sent)) 
train(BernNB, ans1, target) #training the classifier for answers vs. non-answers

# #csvfile=files.upload()
# csvfile=open('rev02.csv', 'rt', newline='') #Open file from class 2 with all responses
# csvfile=files.upload()
# csvfile=open('rev03.csv', 'rt', newline='') #Open file from class 3 with all responses
#csvfile=files.upload()
#csvfile=open('rev04.csv', 'rt', newline='') #Open file from class 4 with all responses
# csvfile=files.upload()
# csvfile=open('rev05.csv', 'rt', newline='') #Open file from class 5 with all responses
# csvfile=files.upload()
# csvfile=open('rev06.csv', 'rt', newline='') #Open file from class 6 with all responses
# csvfile=files.upload()
# csvfile=open('rev07.csv', 'rt', newline='') #Open file from class 7 with all responses
csvfile=files.upload()
csvfile=open('rev08.csv', 'rt', newline='') #Open file from class 8 with all responses
file = csv.reader(csvfile, delimiter=',')
resp=[] #the student answers, each answer in a row
for row in file:
    resp.append(str(row))

csvfile.close()  

classarr=BernNB.predict(resp) #apply the trained filter to filter out non-answers

ans=[]
nonans=[]
for i in range(len(resp)):
    if classarr[i]=='1':
        nonans.append(resp[i]) #array of non-answers
    if classarr[i]=='3':
        ans.append(resp[i])  #array of answers
from sklearn.cluster import KMeans # k-means clustering part of code
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
import string
#from string import maketrans

translator = str.maketrans('', '', string.punctuation)

import nltk
nltk.download('punkt')
#def process_text(text, stem=True):
def process_text(text):
    """ Tokenize text and stem words removing punctuation """
    text = text.translate(translator)
    tokens = word_tokenize(text)
#    if stem:
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(t) for t in tokens]
 
    return tokens
#TFIDF
#vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,1), 
                             #min_df=4, max_df=0.8) #Nov 16: can play with ngram range, min_df, max_df
vectorizer = TfidfVectorizer(stop_words='english', lowercase=True,
         tokenizer=process_text, strip_accents=None, ngram_range=(1,3), 
        min_df=20, max_df=0.4) # Version with process text
X = vectorizer.fit_transform(ans) #turn responses into matrix
#X = vectorizer.fit_transform(pro_sents) #why doesn't this work??
# K-means clustering
true_k = 10 #Nov. 16: Can change number of clusters
kmodel = KMeans(n_clusters=true_k, init='k-means++', max_iter=50000)
kmodel.fit(X)

order_centroids = kmodel.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
labels = kmodel.labels_
centroids = kmodel.cluster_centers_


from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(X)
centdist = 1 -cosine_similarity(X,centroids)

# affinity propagation clustering I tried
# from numpy import unique
# from numpy import where
# from sklearn.datasets import make_classification
# from sklearn.cluster import AffinityPropagation
# from matplotlib import pyplot
# # define the model
# model = AffinityPropagation(damping=0.9)
# # fit the model
# model.fit(X)
# # assign a cluster to each example
# yhat = model.predict(X.toarray())

# # agglomerative clustering I tried
# # from numpy import unique
# # from numpy import where
# # from sklearn.datasets import make_classification
# # from sklearn.cluster import AgglomerativeClustering
# # from matplotlib import pyplot
# # # define dataset
# # model = AgglomerativeClustering(n_clusters=2)
# # # fit model and predict clusters

# birch clustering I tried
# from numpy import unique
# from numpy import where
# from sklearn.datasets import make_classification
# from sklearn.cluster import Birch
# from matplotlib import pyplot
# # define the model
# model = Birch(threshold=0.01, n_clusters=10)
# # fit the model
# model.fit(X)


numex=10 # number of example statements to print

# printing out results
#csvfile= open('Output8.csv', 'wb')
#file=csv.writer(csvfile, delimiter=',', dialect='excel')
print()
numstat=[]
for i in range(true_k):   
    d=kmodel.transform(X)[:,i] # This gives an array of len(X) distances. 
    print("Cluster %d:" % i),
    numstat.append(sum(labels==i))
    print("# statements: %s" % numstat[i])
    thing=[] #a poorly named vector for the top terms in the cluster
    for ind in order_centroids[i, :5]:
#        print(' %s' % terms[ind]),
        thing.append(terms[ind])
    print("List of words for cluster:")
    print(thing) # bytes(string, 'utf-8')
#    file.writerows(bytes(thing, 'utf-8'))
#    print("Statements:")
        #The indices of the 50 closest to centroid j are
    ansind = numpy.argsort(d)[::][:]
#        print(n)
    j=0
#    for n in range(len(ans)):#range(numpy.int(len(ans)/numstat)): #need to fix this    
#        q=n+numpy.int(j*numstat/(numex+1))
#        if i==labels[ansind[q]]:
#            if j<numex:
#                print(ans[ansind[q]])
#                j=j+1                 
 #   for j in range(numex):
 #       if i==labels[ansind[j]]:
 #           print(ans[ansind[j]])
#    examples=[] 
    print("Example student responses:")
    for n in range(len(ans)): ##switch back to ans
         if j < numex:
            if i==labels[n]:
#                 examples.append(ans[n]) ###switch back to ans
                print(ans[n]) #print example statement that student wrote
                j=j+1
#    print(examples)    
    print()
    print() 
print(numstat)
#csvfile.close() 
#from brandonrose.org/clustering


import os  # for os.path.basename

import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.manifold import TSNE

# t-SNE
tsne = TSNE(n_components=2, random_state=0, method='exact')
pos = tsne.fit_transform(X) #why dist not X; what is dist again?




xs, ys = pos[:, 0], pos[:, 1]
print()
print()


#set up colors per clusters using a dict; from brandonrose.org/clustering
cluster_colors = {0: '#ff0000', 1: '#ff9900', 2: '#ffff00', 3: '#33cc33', 
                  4: '#3333ff', 5: '#6600cc', 6: '#ff00ff', 7: '#00ffff', 
                  8: '#800000', 9: '#003300', 10: '#003366', 11: '#660066', 
                  12: '#663300', 13: '#999966', 14: '#ccccff', 15: '#ff99cc',
                  16: '#ffffcc', 17: '#ccffcc', 18: '#ccffff', 19: '#66ccff'}

#create data frame that has the result of the MDS plus the cluster numbers and titles
#df = pd.DataFrame(dict(x=xs, y=ys, label=clusters)) 

#group by cluster
#groups = df.groupby('label')


# set up plot
fig, ax = plt.subplots(figsize=(10, 6)) # set size
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
from matplotlib import pyplot
#pyplot.scatter(xs,ys)

for i in range(true_k):
    for n in range(len(ans)):
        if i==labels[n]:
            pyplot.scatter(xs[n],ys[n], marker='o', 
                    color=cluster_colors[i])


pyplot.show()

for i in range(true_k):
    for n in range(len(ans)):
        if i==labels[n]:
            ax.plot(xs[n],ys[n], marker='o', linestyle='', ms=12, 
                    color=cluster_colors[i], mec='none')
            ax.set_aspect('auto')
            ax.tick_params(\
                           axis= 'x',          # changes apply to the x-axis
                           which='both',      # both major and minor ticks are affected
                           bottom='off',      # ticks along the bottom edge are off
                           top='off',         # ticks along the top edge are off
                           labelbottom='off')
            ax.tick_params(\
                           axis= 'y',         # changes apply to the y-axis
                           which='both',      # both major and minor ticks are affected
                           left='off',      # ticks along the bottom edge are off
                           top='off',         # ticks along the top edge are off
                           labelleft='off')
