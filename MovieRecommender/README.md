### I. Dataset: https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset?resource=download
#### **Because the ratings.csv is too big for training model in jupyter, therefore, we only use first 500001 rows of the data.

#### ![image](https://github.com/dino-3007/Data-Science/assets/109076114/d976d05d-b1c5-4e13-9d05-5184d96ce470)

### II. Descriptive Statistics

### III. Methodology:
#### 1. User-based Collaborative Filtering (CF)

##### - Problems: Scalability and Sparsity
##### - Solution:
#### Focusing on the item similarities => Item-based
#### The key motivation behind Item-based CF is that a customer will more likely to buy
#### item that similar or related to the items being bought already.

#### **Item-based recommender algorithms first analyze the ratings matrix to identify different items relationships, and then use those to indirectly compute item recommendations for users => INEFFICIENT (many pairs do not have any common users)

#### **The best solution is select only the pairs of items for thich the similarity can be computed by initially scanning all the items and for all the users that bought an item, identify the other items bought by those customers. Then efficiently compute the similarities only for these item pairs => focus on Item-Item CF recommender systems.

#### 2. Top N-movies Recommender System
#### Using KNN method in order to identify the K closest instances (movies) given one target instance in our data (about movies). 
#### Ranking all the movies based on their corresponding distances and chooses the top K nearest neighbors (movies) in order to recommend those, as the most similar movies, to a user

### III. Top-N Movies Recommender System
##### Input: Rating data, Movie data, Number of top-N recommendations, Title of target movie
##### Output: list of N most similar movie titles
##### Step 1:
##### * Pre-precessing the rating and movie data
##### * Create a mapper between movie id and movie title
##### * Split the data into train and test sets
##### * Transform the rating (train/test) data to a sparse movie-user matrix

##### Step 2:
##### * Fit the model using movie-user sparse matrix (training set)
##### * Use the input title to find matching movie index
#### * Choose the best match from all matching movie indices
#### * Predict the distances of neighbor (movies) from the target with the number of KNN (K equals to number of top recommendations)
#### * Sort these N neighbor indices
#### * Transform these movie indeices back to the movie titles
#### * Print these titles
#### * Use the test set to calculate the prediction accuracy of the model
