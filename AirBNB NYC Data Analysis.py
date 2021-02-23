#!/usr/bin/env python
# coding: utf-8

# #  AirBNB NYC 

# In[1]:


# Import libraries
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.colors import n_colors
from plotly.subplots import make_subplots


# In[2]:


# Read Data
df= pd.read_csv("AB_NYC_2019.csv")
df.head(10)


# In[3]:


df.shape


# In[4]:


df.describe()


# In[5]:


df.dtypes


# ### Data Exploration

# In[6]:


# check null values
df.isnull().sum()


# In[7]:


# explore the 'calculated_host_listings_count' column
df.loc[(df['host_name']=='John') & (df['host_id']==2787)]


# calculated_host_listings_count = total number of listings by a host

# In[8]:


# explore the categories for 'neighbourhood_group'
df.neighbourhood_group.value_counts()


# In[9]:


# explore the categories for 'neighbourhood'
df.neighbourhood.value_counts()


# In[10]:


# explore the categories for 'room_type'
df.room_type.value_counts()


# In[11]:


# remove the first two columns for now as it is not needed
df.drop(columns=['id', 'name'], axis = 1, inplace = True)


# In[12]:


df.head(5)


# In[13]:


df.info()


# In[14]:


# convert the date field
df['last_review'] = pd.to_datetime(df['last_review'], format = '%d-%m-%Y')

df.info()


# In[15]:


df.describe().round(2)


# In[16]:


# check null values
df.isnull().sum()


# ### Data Cleaning & Processing

# In[17]:


# check if we can impute the missing host_name values from their other listings
df.loc[df['host_id'].isin(df[df['host_name'].isna()].host_id)]


# seems like those who have multiple listings, still have the host_name missing for those too. We assume its for privacy. 'host_id' column can be used instead for identification

# In[18]:


# drop 'host_name' column as we can use 'host_id' for identification/uniqueness

df.drop(columns = 'host_name' , axis = 1 , inplace = True)


# In[19]:


# investigate on null values for 'last_review' & 'reviews_per_month'
df[(df['last_review'].isna()) | (df['reviews_per_month'].isna())]


# we can see that the rows that have null for 'last_review', also has null for 'reviews_per_month'

# In[20]:


# check if all these entries have 0 reviews
df[(df['last_review'].isna()) | (df['reviews_per_month'].isna())].number_of_reviews.value_counts()


# In[21]:


# for NA 'reviews_per_month', we can impute it with '0' as these listings didnt get any reviews yet
df.reviews_per_month.fillna(value = 0, inplace = True)


# In[22]:


# Min of last_review
df.last_review.min()


# In[23]:


# Max of last_review
df.last_review.max()


# In[24]:


# to deal with NA in 'last_review', we change the column to a categorical variable (by year), and those that have NA will be categorised as 'Never', as they did not get any reviews yet
df.last_review = df.last_review.dt.year.astype('object')
df.last_review.fillna(value = 'Never', inplace = True)


# In[25]:


# check 'last_review' categories
df.last_review.value_counts()


# In[26]:


# confirm all null values are dealt with
df.isnull().sum()


# In[27]:


df.head(5)


# In[28]:


df.describe()


# In[29]:


# Check availability_365 
df.availability_365.value_counts()


#  0 days availability can possibly refer to the listing not being available at the moment, so we can keep this data

# In[30]:


# check value for minimum_nights

df.minimum_nights.value_counts()


# In[31]:


# Visualising minimum_nights

fig = px.box(df, y="minimum_nights")
fig.show()


# In[32]:


#check how many listings have minimum nights set to more than a year
df.loc[df['minimum_nights'] > 365]


# Given that having minimum nights as more than 365 days is extremely rare and could be invalid at times as well, we can remove them. Furthermore, its only a few entries, so it should not affect our data.
# 
# 
# Therefore, we will only consider listings that have minimum nights set to a year or less.

# In[33]:


df = df.loc[df['minimum_nights'] <= 365]


# In[34]:


# resetting the count for listings, as a few rows were removed
df['calculated_host_listings_count'] = df.groupby('host_id')['host_id'].transform('count')
# rest index as well
df.reset_index(inplace = True)


# In[35]:


# extreme outliers have been removed, while the moderate outliers are kept, as they are valid but uncommon cases
fig = px.box(df, y="minimum_nights")
fig.show()


# In[36]:


# now things look better
df.describe().round(2)


# ## Data Engineering & Exploratory Data Analysis (EDA)¶
# 

# In[37]:


# Neighbouhood groups
fig = px.scatter_mapbox(df, lat = "latitude", lon = "longitude", color = "neighbourhood_group", size = "calculated_host_listings_count", size_max = 30, opacity = .70, zoom = 12)
fig.layout.mapbox.style = 'carto-positron'
fig.update_layout(title_text = 'NYC Neighbourhood groups with their total listings', height = 750)
fig.show()


# In[38]:


fig = px.scatter_mapbox(df, lat = "latitude", lon = "longitude", color = "neighbourhood", size = "price", size_max = 30, opacity = .70, zoom = 12)
fig.layout.mapbox.style = 'carto-positron'
fig.update_layout(title_text = 'NYC Neighbourhood', height = 750)
fig.show()


# The above chart highlights the neighbourhood comparison. We can state that the popular neighbourhoods are too expensive for certain tourists, they might prefer staying at Airbnbs in neighbourhoods that are comparatively cheaper and closer to locations like Manhattan.

# In[39]:


# Types of rooms according to neighbourhood groups
labels = df.room_type.value_counts().index
values = df.room_type.value_counts().values
fig = px.pie(df, values=values, names=labels, title = "Airbnb Rooms According to Neighbourhood Group")
fig.show()


# Here, we notice that not many prefer shared rooms. Families of more than 4 people might choose an entire home/apartment over a single room. Since the percentage of an entire home/apartments is greater, we can conclude that majority of tourists are a family of more than 2 people and the other portion, a private room would be for couples and single travellers.

# In[40]:


# Availability_365
fig = px.box(df, x = "room_type", y = "availability_365", color = 'room_type')

fig.update_layout(title_text = 'Room type with availability', height = 750)
fig.show()


# #### Price of Each Neighbourhood

# In[41]:


df2 = df.loc[(df['price']<300) & (df['reviews_per_month']<10)]
title = 'Price relation to number of review per month for Properties under $300'
fig = px.scatter(df2, x= 'reviews_per_month', y= 'price',size = 'price', title = title)
fig.show()


# In[42]:


fig = px.bar(df, x= df.number_of_reviews[:50], y = df.neighbourhood[:50] )
fig.show()


# For low price airbnbs, there are some reviews. However, if the prices were high here, we'd see less reviews per month. Not very helpful to have such little reviews. Thus, we may conclude that there is no strong relationship between the reviews per month and price.

# #### 10 Most & Least Expensive Airbnb Neighbourhoods
# 
# 

# In[43]:


df3 = df.dropna(subset=['price']).groupby("neighbourhood")[["neighbourhood","price"]].agg("mean").sort_values(by='price',ascending= False).rename(index = str, columns = {'price':'Average price/night'}).head(10)
fig = px.bar(df3)
fig.show()


# In[44]:


df3 = df.dropna(subset=['price']).groupby("neighbourhood")[["neighbourhood","price"]].agg("mean").sort_values(by='price',ascending= False).rename(index = str, columns = {'price':'Average price/night'}).tail(10)
fig= px.bar(df3)
fig.show()


# There is a sharp drop in prices (around $200) after Tribeca. Rest both charts seem to have similar prices respectively. Tribeca being one of the fanciest localities of New York with trendy boutiques and restaurants alongside Washington Market Park and Hudson River Park,it draws tourists from all around the world. The old industrial buildings turned residential lofts are picture perfect locations for many people.

# In[45]:


# inspect the price differences for each borough
avg_roomtype_cost = df.groupby('room_type').price.median()
top_price = df.groupby(['neighbourhood_group', 'room_type']).median().sort_values(by = 'price', ascending = False).reset_index()

fig = px.line(top_price, x = 'neighbourhood_group', y = 'price', color = 'room_type',color_discrete_sequence = px.colors.colorbrewer.Paired)


fig.show()


# In[46]:



fig = px.parallel_coordinates(df, color = "calculated_host_listings_count",
                              dimensions = ["availability_365","minimum_nights","calculated_host_listings_count","number_of_reviews","price"],
                              labels={"availability_365":"Availability","minimum_nights":"Minimum Nights","calculated_host_listings_count":"Listing Count","number_of_reviews":"Reviews Count","price": "Price"},
                              color_continuous_scale=px.colors.diverging.Tealrose, 
                              color_continuous_midpoint=2)
fig.show()


# In[47]:


df2 = df[df.price < 500]
fig = px.violin(df2,x= 'neighbourhood_group', y = 'price', box = True, color = "neighbourhood_group", title = "Density and distribution of prices for each neighberhood_group'")
fig.show()


# Great, with a statistical table and a violin plot we can definitely observe a couple of things about distribution of prices for Airbnb in NYC boroughs. First, we can state that Manhattan has the highest range of prices for the listings with $150 price as average observation, followed by Brooklyn with \$90 per night. Queens and Staten Island appear to have very similar distributions, Bronx is the cheapest of them all. This distribution and density of prices were completely expected; for example, as it is no secret that Manhattan is one of the most expensive places in the world to live in, where Bronx on other hand appears to have lower standards of living.

# # Consulation
# 
# ### Focus on Neighbourhoods/Neighbourhood_Group :
# Manhattan and Brooklyn have the most listings & customers, Staten Island has the busiest hosts, particularly in Silver Lake, Eltingville and Richmondtown. This is due to less competition, very low costs compared to other boroughs and less minimum night requirements which drives more customers.
# 
# ### Focus on Room-Types & Prices :
#  Private rooms and Entire homes are the most common listings while Shared rooms are much less.
#  Entire homes normally cost around 160 dollars
#  Private rooms cost around 70 dollars
#  Shared rooms cost around 45 dollars
#  Manhattan has the most expensive listings while Staten Island has the cheapest listings.
#  
# ### Focus on Review Rate :
# Althoough Manhattan has a lot of private rooms & entire homes available, many people tend to opt for the   shared rooms, possibly due to lower cost and travelling alone.
# Entire homes seem to be the go to option for most visitors in Staten Island and Bronx.
# There is no significant difference for reviews_per_month between hosts of 'shared rooms' and 'private rooms' (with a 95% confidence)
# 
