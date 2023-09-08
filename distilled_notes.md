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
$
\text{IDF}(t) = \log\left(\frac{1+ \text{Number of Documents}}{1 + \text{Number of Documents containing term \(t\)}}\right) + 1   
$   
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

3\. The similarity matrix obtained using linear kernel in combination with TF-IDF serves as a measurement to quantify the similarity between different documents. The matrix is symmetric and all the diagonal elements are equal to 1 as each document is perfectly similar to itself. The similarity between vectors is often obtained using Cosine Similarity ($ \cos(\theta) = \frac{\vec{A} \cdot \vec{B}}{||\vec{A}|| \times ||\vec{B}||}
\ $) which is the dot product of the vectors divided by their Euclidean norms of magnitudes. In the example above, since the TF-IDF scores are normalized to 1, the similarity matrix has only binary elements as presented below:

``` python
array([[1., 0., 1., 0., 0.],
       [0., 1., 0., 1., 0.],
       [1., 0., 1., 0., 0.],
       [0., 1., 0., 1., 0.],
       [0., 0., 0., 0., 1.]])
```