# Assignment No_02
# Name:Dhamone Siddhesh Dipak
# Roll No : 18
# Title: "Implementation of Bag of Words using Gensim"

# importing libraries
import gensim
from gensim.utils import simple_preprocess
from gensim import corpora

# get input
inp = ["""Data science combines math and statistics, 
       specialized programming, advanced analytics
       , artificial intelligence (AI), and machine learning with specific subject 
       matter expertise to uncover actionable insights hidden in an organization's data."""]

# tokens from input
tokens = []
for line in inp[0].split('.'):
    tokens.append(simple_preprocess(line, deacc=True))

# store into g_dict
g_dict = corpora.Dictionary(tokens)

# Count number of tokens
print("The dictionary has: " + str(len(g_dict)) + " tokens")
print(g_dict.token2id)
print("\n")

# Bag of Words
bow =[g_dict.doc2bow(t, allow_update = True) for t in tokens]
print("Bag of Words : ", bow)

#Output::

#The dictionary has: 28 tokens
#{'actionable': 0, 'advanced': 1, 'ai': 2, 'an': 3, 'analytics': 4, 'and': 5, 
# 'artificial': 6, 'combines': 7, 'data': 8, 'expertise': 9, 'hidden': 10, 'in': 11,
#  'insights': 12, 'intelligence': 13, 'learning': 14, 'machine': 15, 'math': 16,
#  'matter': 17, 'organization': 18, 'programming': 19, 'science': 20, 'specialized': 21,
#  'specific': 22, 'statistics': 23, 'subject': 24, 'to': 25, 'uncover': 26, 'with': 27}


#Bag of Words :  [[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 2), (6, 1), (7, 1), (8, 2),
#  (9, 1), (10, 1), (11, 1), (12, 1), (13, 1), (14, 1), (15, 1), (16, 1), (17, 1), (18, 1),
#  (19, 1), (20, 1), (21, 1), (22, 1), (23, 1), (24, 1), (25, 1), (26, 1), (27, 1)], []]