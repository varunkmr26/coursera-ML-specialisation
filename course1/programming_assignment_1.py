
# coding: utf-8

# In[1]:

import graphlab


# In[2]:

sales = graphlab.SFrame('home_data.gl/')


# In[3]:

train_data,test_data = sales.random_split(.8,seed=0)


# In[5]:

graphlab.canvas.set_target('ipynb')
sales.show(view='BoxWhisker Plot', x='zipcode', y='price')


# In[6]:

#From here we can see that the zipcode of Seattle with the highest average house sale price is '98039'


# In[7]:

#For the first question
zipcode = sales[sales['zipcode']=='98039']
answer1 = zipcode['price'].mean()
answer1


# In[13]:

#For second Question
filter_ = sales[(sales['sqft_living'] > 2000) & (sales['sqft_living'] <= 4000)]
answer2 = float(filter_.shape[0])/sales.shape[0]
answer2


# In[19]:

#For third Question
my_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode']
advanced_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode','condition', 'grade',
                    'waterfront', 'view', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated','lat', 
                     'long','sqft_living15', 'sqft_lot15']

my_features_model = graphlab.linear_regression.create(train_data,target='price',features=my_features,validation_set=None)
my_advanced_model = graphlab.linear_regression.create(train_data,target='price',features=advanced_features,validation_set=None)

answer3 = my_features_model.evaluate(test_data)['rmse'] - my_advanced_model.evaluate(test_data)['rmse']
answer3

