
# coding: utf-8

# # Regression Week 1: Simple Linear Regression

# In this notebook we will use data on house sales in King County to predict house prices using simple (one input) linear regression. You will:
# * Use graphlab SArray and SFrame functions to compute important summary statistics
# * Write a function to compute the Simple Linear Regression weights using the closed form solution
# * Write a function to make predictions of the output given the input feature
# * Turn the regression around to predict the input given the output
# * Compare two different models for predicting house prices
# 
# In this notebook you will be provided with some already complete code as well as some code that you should complete yourself in order to answer quiz questions. The code we provide to complte is optional and is there to assist you with solving the problems but feel free to ignore the helper code and write your own.

# # Fire up graphlab create

# In[1]:

import graphlab


# # Load house sales data
# 
# Dataset is from house sales in King County, the region where the city of Seattle, WA is located.

# In[2]:

sales = graphlab.SFrame('kc_house_data.gl/')


# # Split data into training and testing

# We use seed=0 so that everyone running this notebook gets the same results.  In practice, you may set a random seed (or let GraphLab Create pick a random seed for you).  

# In[16]:

train_data,test_data = sales.random_split(.8,seed=0)
print 'Training data length: ' + str(len(train_data))
print 'Test data length: ' + str(len(test_data))


# # Useful SFrame summary functions

# In order to make use of the closed form solution as well as take advantage of graphlab's built in functions we will review some important ones. In particular:
# * Computing the sum of an SArray
# * Computing the arithmetic average (mean) of an SArray
# * multiplying SArrays by constants
# * multiplying SArrays by other SArrays

# In[17]:

# Let's compute the mean of the House Prices in King County in 2 different ways.
prices = sales['price'] # extract the price column of the sales SFrame -- this is now an SArray

# recall that the arithmetic average (the mean) is the sum of the prices divided by the total number of houses:
sum_prices = prices.sum()
num_houses = prices.size() # when prices is an SArray .size() returns its length
avg_price_1 = sum_prices/num_houses
avg_price_2 = prices.mean() # if you just want the average, the .mean() function
print "average price via method 1: " + str(avg_price_1)
print "average price via method 2: " + str(avg_price_2)


# As we see we get the same answer both ways

# In[18]:

# if we want to multiply every price by 0.5 it's a simple as:
half_prices = 0.5*prices
# Let's compute the sum of squares of price. We can multiply two SArrays of the same length elementwise also with *
prices_squared = prices*prices
sum_prices_squared = prices_squared.sum() # price_squared is an SArray of the squares and we want to add them up.
print "the sum of price squared is: " + str(sum_prices_squared)


# Aside: The python notation x.xxe+yy means x.xx \* 10^(yy). e.g 100 = 10^2 = 1*10^2 = 1e2 

# # Build a generic simple linear regression function 

# Armed with these SArray functions we can use the closed form solution found from lecture to compute the slope and intercept for a simple linear regression on observations stored as SArrays: input_feature, output.
# 
# Complete the following function (or write your own) to compute the simple linear regression slope and intercept:

# In[19]:

def simple_linear_regression(input_feature, output):
    # compute the sum of input_feature and output
    sum_input = input_feature.sum()
    sum_output = output.sum()
    # compute the product of the output and the input_feature and its sum
    sum_product = (input_feature * output).sum()
    # compute the squared value of the input_feature and its sum
    sum_input_squared = (input_feature * input_feature).sum()
    # use the formula for the slope
    n = input_feature.size()
    num = sum_product - ((sum_input*sum_output)/n)
    den = sum_input_squared-((sum_input*sum_input)/n)
    
    slope = num/den
    # use the formula for the intercept
    intercept = sum_output - slope * sum_input
    intercept = intercept/n
    
    return (intercept, slope)


# We can test that our function works by passing it something where we know the answer. In particular we can generate a feature and then put the output exactly on a line: output = 1 + 1\*input_feature then we know both our slope and intercept should be 1

# In[20]:

test_feature = graphlab.SArray(range(5))
test_output = graphlab.SArray(1 + 1*test_feature)
(test_intercept, test_slope) =  simple_linear_regression(test_feature, test_output)
print "Intercept: " + str(test_intercept)
print "Slope: " + str(test_slope)


# Now that we know it works let's build a regression model for predicting price based on sqft_living. Rembember that we train on train_data!

# In[21]:

sqft_intercept, sqft_slope = simple_linear_regression(train_data['sqft_living'], train_data['price'])

print "Intercept: " + str(sqft_intercept)
print "Slope: " + str(sqft_slope)


# # Predicting Values

# Now that we have the model parameters: intercept & slope we can make predictions. Using SArrays it's easy to multiply an SArray by a constant and add a constant value. Complete the following function to return the predicted output given the input_feature, slope and intercept:

# In[22]:

def get_regression_predictions(input_feature, intercept, slope):
    # calculate the predicted values:
    predicted_values = input_feature * slope + intercept
    return predicted_values


# Now that we can calculate a prediction given the slope and intercept let's make a prediction. Use (or alter) the following to find out the estimated price for a house with 2650 squarefeet according to the squarefeet model we estiamted above.
# 
# **Quiz Question: Using your Slope and Intercept from (4), What is the predicted price for a house with 2650 sqft?**

# In[23]:

my_house_sqft = 2650
estimated_price = get_regression_predictions(my_house_sqft, sqft_intercept, sqft_slope)
print "The estimated price for a house with %d squarefeet is $%.2f" % (my_house_sqft, estimated_price)


# # Residual Sum of Squares

# Now that we have a model and can make predictions let's evaluate our model using Residual Sum of Squares (RSS). Recall that RSS is the sum of the squares of the residuals and the residuals is just a fancy word for the difference between the predicted output and the true output. 
# 
# Complete the following (or write your own) function to compute the RSS of a simple linear regression model given the input_feature, output, intercept and slope:

# In[24]:

def get_residual_sum_of_squares(input_feature, output, intercept, slope):
    # First get the predictions
    predictions = get_regression_predictions(input_feature, intercept, slope)
    # then compute the residuals (since we are squaring it doesn't matter which order you subtract)
    residuals = output - predictions
    # square the residuals and add them up
    RSS = (residuals * residuals).sum()
    return(RSS)


# Let's test our get_residual_sum_of_squares function by applying it to the test model where the data lie exactly on a line. Since they lie exactly on a line the residual sum of squares should be zero!

# In[25]:

print get_residual_sum_of_squares(test_feature, test_output, test_intercept, test_slope) # should be 0.0


# Now use your function to calculate the RSS on training data from the squarefeet model calculated above.
# 
# **Quiz Question: According to this function and the slope and intercept from the squarefeet model What is the RSS for the simple linear regression using squarefeet to predict prices on TRAINING data?**

# In[26]:

rss_prices_on_sqft = get_residual_sum_of_squares(train_data['sqft_living'], train_data['price'], sqft_intercept, sqft_slope)
print 'The RSS of predicting Prices based on Square Feet is : ' + str(rss_prices_on_sqft)


# # Predict the squarefeet given price

# What if we want to predict the squarefoot given the price? Since we have an equation y = a + b\*x we can solve the function for x. So that if we have the intercept (a) and the slope (b) and the price (y) we can solve for the estimated squarefeet (x).
# 
# Comlplete the following function to compute the inverse regression estimate, i.e. predict the input_feature given the output!

# In[27]:

def inverse_regression_predictions(output, intercept, slope):
    # solve output = intercept + slope*input_feature for input_feature. Use this equation to compute the inverse predictions:
    estimated_feature = (output - intercept) / slope
    return estimated_feature


# Now that we have a function to compute the squarefeet given the price from our simple regression model let's see how big we might expect a house that costs $800,000 to be.
# 
# **Quiz Question: According to this function and the regression slope and intercept from (3) what is the estimated square-feet for a house costing $800,000?**

# In[28]:

my_house_price = 800000
estimated_squarefeet = inverse_regression_predictions(my_house_price, sqft_intercept, sqft_slope)
print "The estimated squarefeet for a house worth $%.2f is %d" % (my_house_price, estimated_squarefeet)


# # New Model: estimate prices from bedrooms

# We have made one model for predicting house prices using squarefeet, but there are many other features in the sales SFrame. 
# Use your simple linear regression function to estimate the regression parameters from predicting Prices based on number of bedrooms. Use the training data!

# In[29]:

train_data.head()


# In[30]:

# Estimate the slope and intercept for predicting 'price' based on 'bedrooms'
bedroom_intercept, bedroom_slope = simple_linear_regression(train_data['bedrooms'], train_data['price'])

print 'sqft intercept: ' + str(sqft_intercept)
print 'sqft slope: ' + str(sqft_slope)
print "bedrooms Intercept: " + str(bedroom_intercept)
print "bedrooms Slope: " + str(bedroom_slope)


# # Test your Linear Regression Algorithm

# Now we have two models for predicting the price of a house. How do we know which one is better? Calculate the RSS on the TEST data (remember this data wasn't involved in learning the model). Compute the RSS from predicting prices using bedrooms and from predicting prices using squarefeet.
# 
# **Quiz Question: Which model (square feet or bedrooms) has lowest RSS on TEST data? Think about why this might be the case.**

# In[31]:

# Compute RSS when using bedrooms on TEST data:
rss_prices_on_bedroom = get_residual_sum_of_squares(test_data['bedrooms'], test_data['price'], sqft_intercept, sqft_slope)
print 'The RSS of predicting Prices based on Square Feet is : ' + str(rss_prices_on_bedroom)


# In[32]:

# Compute RSS when using squarefeet on TEST data:
rss_prices_on_sqft = get_residual_sum_of_squares(test_data['sqft_living'], test_data['price'], sqft_intercept, sqft_slope)
print 'The RSS of predicting Prices based on Square Feet is : ' + str(rss_prices_on_sqft)


# In[ ]:



