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

# Initialize the TF-IDF vectorizer and remove English stop words
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