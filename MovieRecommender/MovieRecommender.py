#!/usr/bin/env python
# coding: utf-8

# ## Importing library

# In[1]:


get_ipython().system('pip install fuzzywuzzy')


# In[2]:


import numpy as np
import pandas as pd
import math
import time
import gc # 'garbage collection': controlling and managing the memory usage
import argparse # parsing command-line arguments in Python
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import correlation
# utils import
from fuzzywuzzy import fuzz
# for drawing the histograms
import matplotlib.pyplot as plt
import seaborn as sns
# for creating sparse matrices
import scipy.sparse as sparse
from scipy.sparse import csr_matrix
# ignore warnings
import warnings
warnings.filterwarnings('ignore')


# ## Importing data

# In[3]:


ratings_data = pd.read_csv("~/Project/MovieLens_20M_Dataset/rating.csv")
ratings_data = ratings_data[:500000]
ratings_data_org = ratings_data
movies_data = pd.read_csv("~/Project/MovieLens_20M_Dataset/movie.csv")
df_movies = movies_data[['movieId', 'title']]


# ## Quick glance at the data

# In[4]:


ratings_data_org


# In[5]:


movies_data


# In[6]:


#number of unique users and items
n_users = ratings_data.userId.unique().shape[0]
n_items = ratings_data.movieId.unique().shape[0]
print("Number of unique users: " + str(n_users))
print("Number of unique movies: " +str(n_items))


# ## Preprocessing for Descriptive Statistics

# In[8]:


# combining rating data with movie data
combined_data = pd.merge(ratings_data, movies_data, on='movieId')
combined_data.shape


# In[9]:


# unsorted average ratings per movie
combined_data.groupby('userId')['rating'].mean().head()


# In[10]:


# sorted average ratings per movie
combined_data.groupby('title')['rating'].mean().sort_values(ascending=False).head()


# In[11]:


# sorted average ratings per movie with number of ratings (movies that are both often and highly rated)
combined_data.groupby('title')['rating'].count().sort_values(ascending=False).head()


# In[12]:


# column of average rating of each movie
ratings_avg_count = pd.DataFrame(combined_data.groupby('title')['rating'].mean())
ratings_avg_count.head(10)


# In[13]:


# column of number of ratings of that movie
ratings_avg_count['ratings_count'] = pd.DataFrame(combined_data.groupby('title')['rating'].count())
ratings_avg_count.head(10)


# In[14]:


# percentage of movies with less then 10 ratings
x = ratings_avg_count['ratings_count']
count = 0
for i in range(len(x)):
    if x[i] <= 10:
        count += 1
    perc_rat_10 = count * 100 / len(x)
print(perc_rat_10,"%")


# In[15]:


# calculating the total counts of movies with no rating
df_ratings_count_temp = pd.DataFrame(ratings_data.groupby('rating').size(), columns=['count'])
print(df_ratings_count_temp.head(10))
total_count = n_users * n_items
rating_zero_count = total_count - ratings_data.shape[0]
print('rating_zero_count =', rating_zero_count)


# In[16]:


# including 0 ratings
df_ratings_count = df_ratings_count_temp.append(pd.DataFrame({'count': rating_zero_count}, index=[0.0]),verify_integrity=True,).sort_index()
df_ratings_count

## 'verify_integrity=True' ensures that the resulting DataFrame has unique indices


# In[17]:


# adding log count to make sure 0's are also included
df_ratings_count['log_count'] = np.log(df_ratings_count['count'])
df_ratings_count


# In[18]:


# get rating frequency
df_movies_count = pd.DataFrame(ratings_data.groupby('movieId').size(), columns = ['count'])
df_movies_count


# In[19]:


# pop_threshold should be smaller than act_threshold
# filtering the data (movies having rating count > threshold)

N = 50 #threshold
pop_movies = list(set(df_movies_count.query('count >= @N').index)) #'@N' represents a variable or value to be substituted.
ratings_data = ratings_data[ratings_data.movieId.isin(pop_movies)]
print(ratings_data)


# In[20]:


# get number of ratings given by every user
df_user_count = pd.DataFrame(ratings_data.groupby('userId').size(), columns = ['count'])
df_user_count


# In[21]:


# filtering the data (users having rating count > threshold)

M = 50 #threshold
active_user = list(set(df_user_count.query('count >= @M').index))
ratings_data = ratings_data[ratings_data.userId.isin(active_user)]
print(ratings_data)


# In[22]:


print('Original ratings data: ', ratings_data_org.shape)
print('Ratings data after excluding both unpopular movies and inactive users: ', ratings_data.shape)


# In[23]:


#updating row indices to avoid errors in data splitting process (where last row index = (shape of the data-1))
ratings_data = ratings_data.reset_index(drop=True)
ratings_data


# ## EDA

# In[24]:


# Number of ratings counts
plt.figure(figsize=(8,5))
plt.rcParams['patch.force_edgecolor'] = True
ratings_avg_count['ratings_count'].hist(bins=100,color = 'orange')
plt.xlabel('Number of Ratings', fontweight='bold')
plt.ylabel('Counts', fontweight='bold')
plt.xlim([0,2000])


# In[25]:


# Histogram of movie ratings
plt.figure(figsize=(8,5))
plt.rcParams['patch.force_edgecolor'] = True
ratings_avg_count['rating'].hist(bins=50, color = 'orange')
plt.xlabel('Movie Ratings', fontweight='bold')
plt.ylabel('Counts', fontweight='bold')


# In[26]:


# Scatter plot of ratings and counts corresponding to those ratings
plt.figure(figsize=(8,10))
plt.rcParams['patch.force_edgecolor'] = True
figure = sns.jointplot(x='rating', y='ratings_count', data=ratings_avg_count, kind="scatter", color = 'orange')
plt.subplots_adjust(bottom=0.15, top=0.99, left=0.15)
figure.set_axis_labels("Ratings", "Counts", fontweight='bold')


# In[27]:


# rating frequency of all movies
ax = df_movies_count \
    .sort_values('count', ascending=False) \
    .reset_index(drop=True) \
    .plot(figsize=(8, 5),fontsize=14,color = 'orange',)
ax.set_xlabel("Movies", fontweight='bold')
ax.set_ylabel("Number of ratings", fontweight='bold')


# In[28]:


# plot rating frequency of all movies in log scale
ax = df_movies_count \
    .sort_values('count', ascending=False) \
    .reset_index(drop=True) \
    .plot(figsize=(8, 5),fontsize=12,logy=True,color = 'Orange')
ax.set_xlabel("Movies", fontweight='bold')
ax.set_ylabel("Number of ratings (log scale)", fontweight='bold')


# In[29]:


# plot rating frequency of all users
ax = df_movies_count \
    .sort_values('count', ascending=False) \
    .reset_index(drop=True) \
    .plot(figsize=(8, 5),fontsize=14,color = 'orange')
ax.set_xlabel("Users", fontweight='bold')
ax.set_ylabel("Number of ratings", fontweight='bold')


# ## Creating functions

# In[50]:


# creating User/Item/Rating sparse matrices matrices
def sparsematrix(data):
    # Getting the rating matrix
    n_users = data.userId.unique().shape[0]
    n_items = data.movieId.unique().shape[0]
    users_locations = data.groupby(by=['movieId', 'userId', 'rating']).apply(lambda x: 1).to_dict()
    row, col, value = zip(*(users_locations.keys()))
    map_u = dict(zip(data['movieId'].unique(), range(n_items)))
    map_l = dict(zip(data['userId'].unique(), range(n_users)))
    row_idx = [map_u[u] for u in row]
    col_idx = [map_l[l] for l in col]
    datar = np.array(value)
    sparse_csr = csr_matrix((datar, (row_idx, col_idx)), shape=(n_items, n_users))
    # coo_format_sparse = sparse_csr.tocoo([False])
    # csc_format_sparse = sparse_csr.tocsc([False])
    return(sparse_csr, data)


# In[51]:


# splitting data
def get_data(data,test_size):
    unique_users = sorted(pd.unique(data['userId']))
    unique_items = sorted(pd.unique(data['movieId']))
    n_ratings = data.shape[0]
    test = []
    for userid in unique_users:
        # getting this users rating data
        dat = data[data['userId'] == userid]
        # sorting this users data based on time
        dat = dat.sort_values(['timestamp'], ascending=True)
        # test_size*100% of this users ratings
        num = int(dat.shape[0]*test_size)
        # selecting last num% of all ratings of this user
        indext = np.array(dat[-num:].index) ## '[-num:]': the index values of the last num rows
        test.append(indext)
    # (combining indices of all users) list containing the test element indices
    test = np.concatenate(test, axis=None)
    test_items = np.zeros(n_ratings, dtype=bool)
    test_items[test] = True
    test_df = data[['userId', 'movieId', 'rating']][test_items]
    train_df = data[['userId', 'movieId', 'rating']][~test_items]

    # determining the sparse user-item rating matrices in three formats using above datasets
    train_sparse_csr = sparsematrix(train_df)[0]
    test_sparse_csr = sparsematrix(test_df)[0]

    print("Size of the training set")
    print(len(train_df) / (len(train_df)+len(test_df)))
    print("Size of the test set")
    print(len(test_df) / (len(train_df)+len(test_df)))

    return(train_sparse_csr, test_sparse_csr, train_df, test_df)


# In[52]:


## Matching the movies to the titles (title to lower case)
def fuzzy_mapper(movie_mapper, favorite_movie, bool = True):
    matches = []
    for title, index in movie_mapper.items():
        ratio = fuzz.ratio(title.lower(), favorite_movie.lower()) #calculate the similarity between the current movie and the favourite one
        if ratio>= 50:
            matches.append((title, index, ratio))
    # sorting the matches
    matches = sorted(matches, key = lambda x:x[2])[::-1]
    if not matches:
        print('There are no matches found')
        return
    if bool:
        print('Possible matches: {0}\n'.format([x[0] for x in matches]))
    return matches[0][1]


# In[75]:


## Function making similar movie recommendation
def movie_RS(train_data, movie_mapper, favorite_movie, num_recom, distancesb = False):

    # defining the model: similarity, top N movies (n_jobs = -1: using all processors)
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=num_recom, n_jobs=-1)
    # fitting the model using sparse matrix based on the training data
    model_knn.fit(train_data)
    print('The name of the inserted (input) movie:', favorite_movie)
    # to transform the input movie to index
    index = fuzzy_mapper(movie_mapper, favorite_movie, bool=True)
    # because the first returned neighbor is always the target point itself, we add 1 to num_recom(neigbors)
    distances, indices = model_knn.kneighbors(train_data[index], num_recom + 1)
    rec_movies_indecies = \
        sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
    # transforming back to the movie names
    backward_mapper = {v: k for k, v in movie_mapper.items()}
    print('Recommendations for {}:'.format(favorite_movie))
    print('-----------------------------------------------')
    #list of movie indicies recommended for the target movie id
    rec_movie_ind = []
    if distancesb:
       for i, (index, dist) in enumerate(rec_movies_indecies):
            rec_movie_ind.append(index)
            print('{0}: {1}, with distances of {2}'.format(i + 1, backward_mapper[index], dist))
    else:
        for i, (index, dist) in enumerate(rec_movies_indecies):
            print('{0}: {1}'.format(i + 1, backward_mapper[index], dist))
            rec_movie_ind.append(index)

    # testing
    precisions = []
    n = len(test_df.userId.unique())
    for userid in test_df.userId.unique():
        movieids_peruser = []
        hits = 0
        for movieid in test_df[test_df['userId'] == userid]['movieId']:
            movieids_peruser.append(movieid)
        # is there a match between these movies and the recommended movies
        for elem in movieids_peruser:
            for recom in rec_movie_ind:
                if elem == recom:
                  # number of movies that are both in the recommended and test set for this user
                  hits+=1
        precision = hits/num_recom
        precisions.append(precision)
    print('The model precision with top {0} movie recommendations is: {1}'.format(num_recom,round(sum(precisions)/n,3)))


# ## Getting the sets and sparse matrices

# In[54]:


data_object = get_data(ratings_data,0.3)
train_sparse_csr = data_object[0]
test_sparse_csr = data_object[1]
train_df = data_object[2]
test_df = data_object[3]
movie_user_matrix_train = train_df.pivot(index='movieId', columns='userId', values='rating').fillna(0)
movie_user_matrix_test = test_df.pivot(index='movieId', columns='userId', values='rating').fillna(0)


# In[55]:


data_object


# In[56]:


print(train_df)


# In[57]:


print(test_df)


# ## Experiment

# In[93]:


movie_to_index = {
    movie: i for i, movie in
    enumerate(list(df_movies.set_index('movieId').loc[movie_user_matrix_train.index].title))
}

favorite_movies = ['Babe','Copycat', '(500) Days of Summer ', '¡Three Amigos!', 'Things I Hate About You']
N = [3, 5, 10, 20, 30, 50]
for fav_movie in favorite_movies:
    for num_rec in N:
        print(fav_movie)
        print(num_rec)
        print(movie_RS(train_data = train_sparse_csr,favorite_movie = fav_movie, movie_mapper = movie_to_index, num_recom = num_rec, distancesb = False))


# ## Plotting graph for the first movie

# In[96]:


# Movie 1: Babe
# N = 3 0.025
# N = 5 0.018
# N = 10 0.016
# N = 20 0.02
# N = 30 0.019
# N = 50 0.02

bars = [0.025, 0.018, 0.016, 0.02, 0.019, 0.02]
# r1 = [1, 2, 3, 4, 5]
barWidth = 0.4
r1 = np.arange(len(bars))
r2 = [x + barWidth for x in r1]
plt.bar(r2, bars, color='black', width=barWidth, edgecolor='white')
plt.xlabel('Top-N Recommendation', fontweight='bold')
plt.ylabel('Precision', fontweight='bold')
plt.title('Babe', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars))],['N 3', 'N 5', 'N 10', 'N 20', 'N 30','N 50'], rotation='vertical')
plt.subplots_adjust(bottom=0.23, top=0.92, left=0.15)
plt.legend()
plt.show()


# In[ ]:




