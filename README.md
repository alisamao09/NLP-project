# research-project-1
Classifying students' open-ended responses in large class
## Overview
Open-ended questions can take a huge amount of time to read, especially in a large class of size of around 800 people
In order to reduce the work load of reading these responses for the faculty, we design a set of algorithm based on NLP
Before each class, students would answer the open-ended question:"What was the most challenging or confusing part of this chapter reading and/or pre-class video?"
Arounded 500 students answered this pre-class question

## Main process
* Tested multiple training and test filters to filter out non-answers, with a success rate above 90%
* Pre-processed text data by stemming words, removing stop words and punctuations, making lowercase, and including "tuples"
* Calculated word frequency by TFIDF
* Explored multiple unsupervised clustering algorithms such as K-means, affinity clustering, and GMM etc.
* Projected clusterings to 2D space and printed out N important terms for each cluster

## Acknowledgement
I would like to thank my research project supervisor, Professor Carolyn Sealfon, for the amount of effort devoted, her patience, support, guidance, and also her valuable insight into my confusion. I'm also grateful for the opportunity to work on this fun project.
