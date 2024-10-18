from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
import requests

app = Flask(__name__)

# Load the dataset
df = pd.read_csv(r"D:\abdo\AI\projects\recommender\chatbot\movies_with_tags_rating.csv")

# Initialize DistilBERT model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Clean movie titles
def clean_movie_title(title):
    return re.sub(r'\s*\(.*?\)\s*', '', str(title).lower()).strip()

# Apply title cleaning to the dataset
df['cleaned_title'] = df['title'].apply(clean_movie_title)

# Load precomputed embedding matrices
embedding_features = ['overview', 'genre', 'cast', 'director', 'combined']
for feature in embedding_features:
    matrix = np.load(f'{feature}_embedding_matrix.npy')
    df[f'{feature}_embedding'] = list(matrix)

# Fetch similar movies from TMDb
def fetch_tmdb_recommendations(movie_title):
    api_key = "your_tmdb_api_key"
    search_response = requests.get(f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={movie_title}").json()

    print(f"TMDb search response for '{movie_title}': {search_response}")  # Debugging line

    if 'results' in search_response and search_response['results']:
        movie_id = search_response['results'][0]['id']
        recommendations_response = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}/similar?api_key={api_key}").json()

        genres_response = requests.get(f"https://api.themoviedb.org/3/genre/movie/list?api_key={api_key}").json()
        genre_mapping = {genre['id']: genre['name'] for genre in genres_response['genres']}

        similar_movies = []
        for movie in recommendations_response['results']:
            movie_genres = [genre_mapping.get(genre_id, "Unknown") for genre_id in movie['genre_ids']]
            similar_movies.append({
                'title': movie['title'],
                'genre': ', '.join(movie_genres),
                'rating': movie.get('vote_average', 'N/A')
            })

        return pd.DataFrame(similar_movies).head(3)

    return None

# Recommend similar movies
def recommend_similar_movies(movie_title, df, top_n=3):
    cleaned_movie_title = clean_movie_title(movie_title)

    if cleaned_movie_title in df['cleaned_title'].values:
        movie_index = df[df['cleaned_title'] == cleaned_movie_title].index[0]
        input_embedding = df.loc[movie_index, 'combined_embedding']
        embedding_matrix = np.stack(df['combined_embedding'].values)

        similarities = cosine_similarity([input_embedding], embedding_matrix)[0]
        similar_movies = df.iloc[np.argsort(similarities)[::-1][1:top_n+1]]
        return similar_movies[['title', 'genre', 'rating']]
    
    return fetch_tmdb_recommendations(movie_title)

# Flask route to accept chatbot input and return recommendations
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    
    # Modify regular expression to handle extra words after the movie title
    movie_title = re.search(r"(?:love|like|watch)\s+([a-zA-Z\s]+)", user_message, re.IGNORECASE)
    
    if movie_title:
        # Strip common phrases like "what should I watch next"
        movie_title = movie_title.group(1).strip()
        movie_title = re.sub(r"\b(what should i watch next)\b", "", movie_title, flags=re.IGNORECASE).strip()

        recommendations = recommend_similar_movies(movie_title, df)

        if recommendations is not None:
            return jsonify(recommendations.to_dict(orient='records'))
        else:
            return jsonify({'message': f"No recommendations found for '{movie_title}'."})
    
    return jsonify({'message': 'Please provide a valid movie title in your query.'})

if __name__ == '__main__':
    app.run(debug=True)
