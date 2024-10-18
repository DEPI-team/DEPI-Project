from flask import Flask, request, jsonify
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests
from transformers import AutoTokenizer, AutoModel
import re

app = Flask(__name__)

# Load the dataset
df = pd.read_csv("movies_with_tags_rating.csv")

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
    api_key = "ddad317e776c8ec2f92ec52efe9d34f5"  
    search_response = requests.get(f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={movie_title}").json()

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
        
        similar_movies_df = pd.DataFrame(similar_movies)

        # Get the top 3 movies
        top_movies_df = similar_movies_df.head(3)

        return top_movies_df
    
    print(f"No results found for '{movie_title}' in TMDb. Response: {search_response}")
    return None

# Recommend similar movies
def recommend_similar_movies(movie_title, top_n=3):
    cleaned_movie_title = clean_movie_title(movie_title)

    if cleaned_movie_title in df['cleaned_title'].values:
        movie_index = df[df['cleaned_title'] == cleaned_movie_title].index[0]

        input_embedding = df.loc[movie_index, 'combined_embedding']
        embedding_matrix = np.stack(df['combined_embedding'].values)

        similarities = cosine_similarity([input_embedding], embedding_matrix)[0]
        similar_movies = df.iloc[np.argsort(similarities)[::-1][1:top_n+1]]
        return similar_movies[['title', 'genre', 'rating']].to_json(orient='records')
        
    else:
        print(f"Movie '{movie_title}' not found in the dataset. Fetching recommendations from TMDb...")
        recommendations = fetch_tmdb_recommendations(movie_title)
        if recommendations is not None and not recommendations.empty:
            return recommendations.to_json(orient='records')
        else:
            return jsonify([])

@app.route('/recommend', methods=['GET'])
def recommend():
    movie_title = request.args.get('movie_title')
    top_n = int(request.args.get('top_n', 3))
    return recommend_similar_movies(movie_title, top_n)

if __name__ == '__main__':
    app.run(debug=True)