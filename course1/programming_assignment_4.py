
# coding: utf-8

# In[1]:

import graphlab


# In[13]:

song_data = graphlab.SFrame('song_data.gl/')


# In[8]:

graphlab.canvas.set_target('ipynb')


# In[3]:

kanya_west = song_data[song_data['artist']=='Kanye West']['user_id'].unique()
kanya_west


# In[4]:

foo_fighters = song_data[song_data['artist']=='Foo Fighters']['user_id'].unique()
foo_fighters


# In[5]:

taylor_swift = song_data[song_data['artist']=='Taylor Swift']['user_id'].unique()
taylor_swift


# In[6]:

lady_gaga = song_data[song_data['artist']=='Lady GaGa']['user_id'].unique()
lady_gaga


# In[15]:

artist_count = song_data.groupby(key_columns='artist', operations={'total_count': graphlab.aggregate.SUM('listen_count')})


# In[16]:

artist_count.head()


# In[20]:

artist_count = artist_count.sort('total_count')


# In[21]:

most_popular = artist_count['artist'][-1]
most_popular


# In[22]:

least_popular = artist_count['artist'][0]
least_popular


# In[23]:

train_data,test_data = song_data.random_split(.8,seed=0)


# In[24]:

personalized_model = graphlab.item_similarity_recommender.create(train_data,
                                                                user_id='user_id',
                                                                item_id='song')


# In[25]:

subset_test_users = test_data['user_id'].unique()[0:10000]


# In[27]:

recommendations = personalized_model.recommend(subset_test_users,k=1)


# In[30]:

recommend_count = recommendations.groupby(key_columns='song', operations={'count': graphlab.aggregate.COUNT()})
recommend_count.sort('count',ascending=False)


# In[31]:

most_recommended_song = 'Undo - Björk'
most_recommended_song

