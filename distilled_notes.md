# Recommender System
A curated collection of essential notes on recommender systems, distilled for quick insights and easy understanding.

# Methodologies
For recommender system two well established techniques include: content based filtering and collaborative filtering. In former, the recommendation is made based on the contents similarity and in the latter based on the behavior of other users.

## Content based filtering
In this approach, if the users has shown interest in product A, we can extract the features from this product and find similar products. The most similar products could then be recommended to the user. Below is a simple example to understand how this approach works:

```python
# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Create a sample DataFrame with movie titles and their genres
movies = pd.DataFrame({
    'title': ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E'],
    'genre': ['Action', 'Comedy', 'Action', 'Comedy', 'Drama']
})

# Initialize the (Term Frequency Inverse Document Frequency) TF-IDF vectorizer and remove English stop words
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Transform the 'genre' column into TF-IDF vectors
tfidf_matrix = tfidf_vectorizer.fit_transform(movies['genre'])

# Compute the cosine similarity matrix from the TF-IDF vectors
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to make movie recommendations based on genre
def recommend(title):
    # Find the index of the movie that matches the title
    idx = movies.index[movies['title'] == title].tolist()[0]
    
    # Enumerate over the similarity scores of all movies and make a list
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort the movies based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the scores for 2 most similar movies; skip the input movie itself (hence [1:3])
    sim_scores = sim_scores[1:3]
    
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    # Return the top 2 most similar movies
    return movies['title'].iloc[movie_indices]

# Test the function with 'Movie A'
print(recommend('Movie A'))
```
Below are a few notes about the code snippet above:   

1\. The initialization step determines the parameters that should be accounted for while doing the feature extraction during the fitting step. In this example it sets `stop_words=english` to avoid common english words during feature extraction.

2\. For move genres the TF-IDF is calculated assuming that each move is a single document. Hence, TF is the frequency of each word within the document which is 1 for all the movies and IDF represents the rarity of each genre across all 5 movies which is calculated using the following equation:    

IDF(t) = log((1+ Number of Documents) / (1 + Number of Documents containing term (t))) + 1

Hence the IDF for action and comedy movies would be ln(6/3)+1 = 1.693 (i.e., TF-IDF = 1*1.693 = 1.693) and for drama movie would be ln(6/2)+1 = 2.0986 (i.e., TF-IDF = 1*2.0986 = 2.0986). As such, since the drama genre is relative rarer, it has a higher score.    
`Note` that The Scikit-learn library normalizes the TF-IDF scores and since each document contains only one word, the score for each document would be equal to 1 after normalization. Below is the code snippet to check the TF-IDF scores for each move genre:

```python 
dense_matrix = tfidf_matrix.todense()
df = pd.DataFrame(dense_matrix, columns=tfidf_vectorizer.get_feature_names_out())
print(df)
   action  comedy  drama
0     1.0     0.0    0.0
1     0.0     1.0    0.0
2     1.0     0.0    0.0
3     0.0     1.0    0.0
4     0.0     0.0    1.0
```
As presented in the code snippet above, the Scikit-learn stores the TF-IDF scores in `sparse` format which doesn't include zeros for storage efficiency. In order to visualize the score, the score matrix much first be converted into a dense format which also includes zeros as printed above in a form of a pandas dataset. It is also worth noting that in the above example there are 3 unique genres (i.e., tokens) across all the movies (i.e., documents), hence each TF-IDF vector which represents a movie has 3 elements.

3\. The similarity matrix obtained using linear kernel in combination with TF-IDF serves as a measurement to quantify the similarity between different documents. The matrix is symmetric and all the diagonal elements are equal to 1 as each document is perfectly similar to itself. The similarity between vectors is often obtained using Cosine Similarity (Cos(theta) = A.B / (||A|| * ||B||)) which is the dot product of the vectors divided by their Euclidean norms (i.e., magnitudes). In the example above, since the TF-IDF scores are normalized to 1, the similarity matrix has only binary elements as presented below:

``` python
array([[1., 0., 1., 0., 0.],
       [0., 1., 0., 1., 0.],
       [1., 0., 1., 0., 0.],
       [0., 1., 0., 1., 0.],
       [0., 0., 0., 0., 1.]])
```

## Collaborative-based filtering
The collaborative-based filtering is applied in applications such as when user 1 liked or purchased product A and B and user 2 liked or purchased product A, B, and C and according to this approach is probably that user A would also be interested in product C. Similar to content-based filtering, let's dive into a simple example in python:

``` python
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Create a sample DataFrame with user IDs, movie IDs, and ratings
ratings = pd.DataFrame({
    'user_id': [1, 1, 2, 2, 3, 3, 4, 4, 5],
    'movie_id': [1, 2, 1, 3, 3, 4, 2, 4, 5],
    'rating': [5, 4, 4, 3, 2, 4, 5, 3, 1]
})

# Create a user-item interaction matrix
interaction_matrix = pd.pivot_table(ratings, index='user_id', columns='movie_id', values='rating')

# Replace NaN with 0, because we will consider the absence of rating as a zero rating
interaction_matrix = interaction_matrix.fillna(0)
print(interaction_matrix,'\n')

# Initialize the Nearest Neighbors model
knn = NearestNeighbors(metric='cosine', algorithm='brute')

# Fit the model on the interaction matrix
knn.fit(interaction_matrix)

# Function to recommend movies for a given user
for user_id in range(1,5):
    # Get the index of the user
    user_index = interaction_matrix.index.get_loc(user_id)

    # Find the nearest neighbors for the given user
    distances, indices = knn.kneighbors(interaction_matrix.iloc[user_index, :].values.reshape(1, -1), n_neighbors=2)
    print(f"for user id {user_id}, the distance: {distances[0]}, and indices are {indices[0]}, ")

    # Get the list of similar users
    similar_users = interaction_matrix.index[indices.flatten()].tolist()

    # Remove the user itself from the list
    similar_users.remove(user_id)

    # Get the list of movies rated by the similar users
    recommended_movies = interaction_matrix.loc[similar_users].mean().sort_values(ascending=False).index.tolist()

    # Remove the movies already rated by the user
    rated_movies = interaction_matrix.columns[interaction_matrix.loc[user_id] > 0].tolist()
    recommended_movies = [m for m in recommended_movies if m not in rated_movies]
    print(f"The recommended movies are {recommended_movies} \n")
  
#Output:

movie_id    1    2    3    4    5
user_id                          
1         5.0  4.0  0.0  0.0  0.0
2         4.0  0.0  3.0  0.0  0.0
3         0.0  0.0  2.0  4.0  0.0
4         0.0  5.0  0.0  3.0  0.0
5         0.0  0.0  0.0  0.0  1.0 

for user id 1, the distance: [0.         0.37530495], and indices are [0 1], 
The recommended movies are [3, 4, 5] 

for user id 2, the distance: [0.         0.37530495], and indices are [1 0], 
The recommended movies are [2, 4, 5] 

for user id 3, the distance: [1.11022302e-16 5.39821007e-01], and indices are [2 3], 
The recommended movies are [2, 1, 5] 

for user id 4, the distance: [2.22044605e-16 4.64328416e-01], and indices are [3 0], 
The recommended movies are [1, 3, 5] 

```

Below are a few notes about the code snippet above:   
1\. Each row in the interaction matrix represents a vector and the similarity between different vectors is calculated using cosine similarity. For instance the distance between vector 1, 2 that represent user_id 1 and 2 is calculated as below:

Cosine Similarity = ((5.0 * 4.0) + (4.0 * 0.0) + (0.0 * 3.0) + (0.0 * 0.0) + (0.0 * 0.0)) / (sqrt(5.0^2 + 4.0^2 + 0.0^2 + 0.0^2 + 0.0^2) * sqrt(4.0^2 + 0.0^2 + 3.0^2 + 0.0^2 + 0.0^2)) ≈ 0.625


The cosine similarity calculates the cosine of the angle between two vectors. Hence, the closer the vectors are, the closer the angle is to 0, making the cosine similarity closer to 1. To convert this into a cosine distance, we subtract the cosine similarity from 1, leading to 1 − 0.625 = 0.375.
