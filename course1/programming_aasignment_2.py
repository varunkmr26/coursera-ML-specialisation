
# coding: utf-8

# In[1]:

import graphlab


# In[2]:

products = graphlab.SFrame('amazon_baby.gl/')


# In[3]:

products['word_count'] = graphlab.text_analytics.count_words(products['review'])


# In[4]:

graphlab.canvas.set_target('ipynb')


# In[5]:

products = products[products['rating'] != 3]


# In[6]:

products['sentiment'] = products['rating'] >=4


# In[7]:

selected_words = ['awesome', 'great', 'fantastic', 'amazing', 'love', 'horrible', 'bad', 'terrible', 'awful', 'wow', 'hate']


# In[13]:

for word in selected_words:
    def count_function(x):
        if word in x:
            return x[word]
        else:
            return 0
    products[word] = products['word_count'].apply(count_function)
products.head()


# In[14]:

word_counts = {}
for word in selected_words:
    word_counts[word] = products[word].sum()
word_counts


# In[15]:

answer1 = 'great'
answer1


# In[16]:

answer2 = 'wow'
answer2


# In[18]:

train_data,test_data = products.random_split(.8, seed=0)


# In[19]:

sentiment_model = graphlab.logistic_classifier.create(train_data,
                                                     target='sentiment',
                                                     features=['word_count'],
                                                     validation_set=test_data)


# In[20]:

selected_words_model = graphlab.logistic_classifier.create(train_data,
                                                     target='sentiment',
                                                     features=selected_words,
                                                     validation_set=test_data)


# In[21]:

sentiment_model.evaluate(test_data, metric='roc_curve')


# In[22]:

sentiment_model.show(view='Evaluation')


# In[23]:

selected_words_model.evaluate(test_data, metric='roc_curve')


# In[24]:

selected_words_model.show(view='Evaluation')


# In[25]:

products['sentiment'].show()


# In[31]:

x = selected_words_model['coefficients'].sort('value')
x


# In[32]:

answer3 = x[-1]
answer3


# In[33]:

answer4 = x[0]
answer4


# In[35]:

answer5 = 0.843
answer5


# In[36]:

answer6 = 0.916
answer6


# In[37]:

answer7 = 0.841
answer7


# In[38]:

diaper_champ_reviews = products[products['name'] == 'Baby Trend Diaper Champ']


# In[39]:

diaper_champ_reviews['predicted_sentiment_sentiment_model'] = sentiment_model.predict(diaper_champ_reviews, output_type='probability')
diaper_champ_reviews = diaper_champ_reviews.sort('predicted_sentiment_sentiment_model', ascending=False)
diaper_champ_reviews.head()


# In[40]:

answer9 = 0.999999937267
answer9


# In[41]:

answer10 = selected_words_model.predict(diaper_champ_reviews[0], output_type='probability')


# In[42]:

answer10


# In[43]:

diaper_champ_reviews[0]['review']

