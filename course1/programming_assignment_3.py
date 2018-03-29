
# coding: utf-8

# In[1]:

import graphlab


# In[2]:

people = graphlab.SFrame('people_wiki.gl/')


# In[3]:

people['word_count'] = graphlab.text_analytics.count_words(people['text'])


# In[4]:

tfidf = graphlab.text_analytics.tf_idf(people['word_count'])
people['tfidf'] = tfidf


# In[5]:

john = people[people['name'] == 'Elton John']


# In[7]:

john[['word_count']].stack('word_count', new_column_name = ['word','count']).sort('count',ascending=False)


# In[8]:

answer1 = 'the,in,and'
answer1


# In[10]:

john[['tfidf']].stack('tfidf',new_column_name=['word','tfidf']).sort('tfidf',ascending=False)


# In[11]:

answer2 = 'furnish,elton,billboard'
answer2


# In[14]:

beckham = people[people['name'] == 'Victoria Beckham']
answer3 = graphlab.distances.cosine(john['tfidf'][0],beckham['tfidf'][0])
answer3


# In[15]:

paul = people[people['name'] == 'Paul McCartney']
answer4 = graphlab.distances.cosine(john['tfidf'][0],paul['tfidf'][0])
answer4


# In[27]:

answer5 = 'Paul McCartney'
answer5


# In[17]:

tfidf_model = graphlab.nearest_neighbors.create(people,features=['tfidf'],label='name',distance='cosine')


# In[18]:

word_count_model = graphlab.nearest_neighbors.create(people,features=['word_count'],label='name',distance='cosine')


# In[19]:

word_count_model.query(john)


# In[20]:

answer6 = 'Cliff Richard'
answer6


# In[21]:

tfidf_model.query(john)


# In[22]:

answer7 = 'Rod Stewart'
answer7


# In[23]:

word_count_model.query(beckham)


# In[24]:

answer8 = 'Mary Fitzgerald (artist)'
answer8


# In[25]:

tfidf_model.query(beckham)


# In[26]:

answer9 = 'David Beckham'
answer9

