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

##############################
#!/usr/bin/env python
# coding: utf-8

# In[12]:


from textblob import TextBlob
import nltk


# In[2]:


get_ipython().system('pip install TextBlob')


# In[4]:


mystr = """The best error message is the one that never shows up.
You Learn More From Failure Than From Success. 
Don’t Let It Stop You. Failure Builds Character.
If You Are Working On Something That You Really Care About, You Don’t Have To Be Pushed. The Vision Pulls You.
The purpose of software engineering is to control complexity, not to create it"""


# In[5]:


blob = TextBlob(mystr)


# In[6]:


type(blob)


# In[7]:


from textblob.blob import BaseBlob


# In[8]:


base_blob = BaseBlob(mystr)


# In[9]:


type(base_blob)


# Word Tokenizing
# 

# In[10]:


blob


# In[14]:


nltk.download('punkt')


# In[15]:


blob.words


# Sentence Tokenizing
# 

# In[16]:


blob.sentences


# In[17]:



# Words tokens
for word_tokens in blob.sentences:
    print(word_tokens.words)


# N-Grams
# 

# In[18]:



# Bi-Gram
for bigram in blob.ngrams(2):
    print(bigram)


# In[19]:


# Tri-Gram
for trigram in blob.ngrams(3):
   print(trigram)


# Part of Speech
# 

# In[20]:


mystr


# In[21]:


blob1 = TextBlob("Hello world this is NLP with TextBlob for text classification.")


# In[24]:


blob1.tags


# In[23]:


get_ipython().system('python -m textblob.download_corpora')


# In[25]:


## Alternative Part of Speech
blob1.pos_tags


# Noun_Phrases

# In[26]:


blob2 = TextBlob("Google is great search engine for finding almost anything ")


# In[27]:



for np in blob2.noun_phrases:
    print(np)


# In[28]:


# ['VB','VBZ','VBP','VBD','VBN','VBG']
for word,verb in blob.tags:
    if verb in ['VB','VBZ','VBP','VBD','VBN','VBG']:
        print(f'{word} => {verb}')


# In[ ]:







