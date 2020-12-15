#!/usr/bin/env python
# coding: utf-8

# In[8]:


import spacy
nlp = spacy.load('en_core_web_sm')


# In[2]:


import nltk
from spacy.lang.en import English


# In[3]:


get_ipython().system('python -m spacy download en_core_web_sm')


# In[4]:


get_ipython().system('python spacy download en')


# In[5]:


get_ipython().system('pip install -U spacy')


# In[6]:


def pos_tagging(s):
    s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)
    words = nltk.word_tokenize(s)
    return nltk.pos_tag(words)


# In[9]:


sp = spacy.load('en_core_web_sm')
sentence = sp("I like to play football.I hated it in my childhood though")
print(sentence.text)


# In[12]:


import spacy
def pos_tagging_s():
    sp = spacy.load('en_core_web_sm')
    sen = sp("I like to play football. I hated it in my childhood though")
    print(sen.text)
    print(sen[1].pos_)
    print(sen[1].tag_) 
    print(spacy.explain(sen[1].tag_))
    for word in sen:
        print("Word:", word.text, "\t","POS Tag:", word.pos_,"\t", "Tag for Word:", word.tag_,"Explanatation:", spacy.explain(word.tag_), "\n")
print(pos_tagging_s())   


# In[ ]:




