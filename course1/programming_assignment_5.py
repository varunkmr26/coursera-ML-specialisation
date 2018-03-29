
# coding: utf-8

# In[1]:

import graphlab


# In[2]:

image_train = graphlab.SFrame('image_train_data/')
image_test = graphlab.SFrame('image_test_data/')


# In[3]:

image_train['label'].sketch_summary()


# In[4]:

answer1 = 'bird'
answer1


# In[6]:

car_frame = image_train[image_train['label'] == 'automobile']
cat_frame = image_train[image_train['label'] == 'cat']
dog_frame = image_train[image_train['label'] == 'dog']
bird_frame = image_train[image_train['label'] == 'bird']


# In[8]:

car_model = graphlab.nearest_neighbors.create(car_frame,features=['deep_features'],
                                             label='id')


# In[9]:

cat_model = graphlab.nearest_neighbors.create(cat_frame,features=['deep_features'],
                                             label='id')


# In[10]:

dog_model = graphlab.nearest_neighbors.create(dog_frame,features=['deep_features'],
                                             label='id')


# In[11]:

bird_model = graphlab.nearest_neighbors.create(bird_frame,features=['deep_features'],
                                             label='id')


# In[12]:

cat_test = image_test[0:1]
cat_test


# In[13]:

def get_images_from_ids(query_result):
    return image_train.filter_by(query_result['reference_label'],'id')


# In[20]:

cat_neighbors =cat_model.query(cat_test)


# In[39]:

cat_neighbors_photos = get_images_from_ids(cat_neighbors)
cat_neighbors_photos['image'].show() #This is answer 2


# In[21]:

dog_neighbors = dog_model.query(cat_test)
dog_neighbors


# In[40]:

dog_neighbors_photos = get_images_from_ids(dog_neighbors)
dog_neighbors_photos['image'].show() #This is answer 3


# In[25]:

cat_distance = cat_neighbors['distance'].mean() 
cat_distance #This is answer 4


# In[26]:

dog_distance = dog_neighbors['distance'].mean()
dog_distance #This is answer 5


# #The answer 6 is 'cat'

# In[27]:

image_test_cat = image_test[image_test['label'] == 'cat']
image_test_dog = image_test[image_test['label'] == 'dog']
image_test_bird = image_test[image_test['label'] == 'bird']
image_test_automobile = image_test[image_test['label'] == 'automobile']


# In[28]:

dog_cat_neighbors = cat_model.query(image_test_dog, k=1)


# In[29]:

dog_dog_neighbors = dog_model.query(image_test_dog, k=1)


# In[30]:

dog_automobile_neighbors = car_model.query(image_test_dog, k=1)


# In[31]:

dog_bird_neighbors = bird_model.query(image_test_dog, k=1)


# In[33]:

dog_distances = graphlab.SFrame({'dog-dog': dog_dog_neighbors['distance'],'dog-cat': dog_cat_neighbors['distance'],'dog-bird': dog_bird_neighbors['distance'],'dog-automobile': dog_automobile_neighbors['distance']})


# In[34]:

dog_distances


# In[35]:

def is_dog_correct(row):
    if ((row['dog-dog'] < row['dog-cat']) and (row['dog-dog'] < row['dog-bird']) and (row['dog-dog'] < row['dog-automobile'])):
        return 1
    else:
        return 0


# In[36]:

dog_distances['is_dog_correct'] = dog_distances.apply(is_dog_correct)


# In[45]:

correct = dog_distances['is_dog_correct'].sum()
answer7 = correct/1000.
answer7


# In[ ]:



