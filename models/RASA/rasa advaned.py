import re
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import requests
from tmdbv3api import TMDb, Movie
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the dataset
df = pd.read_csv(r"D:\abdo\AI\projects\recommender\chatbot\movies_with_tags_rating.csv")

# Initialize RoBERTa model and tokenizer for NER
model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
ner_model = AutoModelForTokenClassification.from_pretrained("roberta-base")
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=tokenizer)

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

# Initialize TMDb API
tmdb = TMDb()
tmdb.api_key = 'ddad317e776c8ec2f92ec52efe9d34f5'
movie = Movie()

# Initialize MovieLens API
movielens_api = "http://api.themoviedb.org/3"

# Extract movie title using RoBERTa NER
def extract_movie_title_roberta(user_message):
    try:
        # Use the NER pipeline directly with the user message
        ner_results = ner_pipeline(user_message)

        if not ner_results:
            return None  # Early exit if no results

        movie_title = ""
        for entity in ner_results:
            if entity['entity'].startswith("B-") or entity['entity'].startswith("I-"):
                if entity['word'].isalpha():  # Only consider alphabetic words
                    movie_title += entity['word'] + " "

        return movie_title.strip() if movie_title else None
    except IndexError as e:
        print(f"Error: {e}")
        return None

# Fetch recommendations from TMDb
def fetch_tmdb_recommendations(movie_title):
    search_response = requests.get(f"https://api.themoviedb.org/3/search/movie?api_key={tmdb.api_key}&query={movie_title}").json()
    if search_response.get('results'):
        movie_id = search_response['results'][0]['id']
        recommendations_response = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}/similar?api_key={tmdb.api_key}").json()

        # Extract genre information for recommendations
        genres_response = requests.get(f"https://api.themoviedb.org/3/genre/movie/list?api_key={tmdb.api_key}").json()
        genre_mapping = {genre['id']: genre['name'] for genre in genres_response['genres']}

        similar_movies = []
        for movie in recommendations_response.get('results', []):
            movie_genres = [genre_mapping.get(genre_id, "Unknown") for genre_id in movie['genre_ids']]
            similar_movies.append({
                'title': movie['title'],
                'genre': ', '.join(movie_genres),
                'rating': movie.get('vote_average', 'N/A')
            })

        return pd.DataFrame(similar_movies).head(3)
    return None

# Fetch recommendations from MovieLens
def fetch_movielens_recommendations(movie_title):
    search_response = requests.get(f"{movielens_api}/search/movie?api_key={tmdb.api_key}&query={movie_title}").json()
    if search_response.get('results'):
        movie_id = search_response['results'][0]['id']
        recommendations_response = requests.get(f"{movielens_api}/movie/{movie_id}/recommendations?api_key={tmdb.api_key}").json()

        similar_movies = []
        for movie in recommendations_response.get('results', []):
            similar_movies.append({
                'title': movie['title'],
                'genre': movie.get('genres', 'N/A'),
                'rating': movie.get('vote_average', 'N/A')
            })

        return pd.DataFrame(similar_movies).head(3)
    return None

# Recommend similar movies using cosine similarity or TMDb/MovieLens fallback
def recommend_similar_movies(movie_title, df, top_n=3):
    cleaned_movie_title = clean_movie_title(movie_title)

    if cleaned_movie_title in df['cleaned_title'].values:
        movie_index = df[df['cleaned_title'] == cleaned_movie_title].index[0]
        input_embedding = df.loc[movie_index, 'combined_embedding']
        embedding_matrix = np.stack(df['combined_embedding'].values)

        # Compute cosine similarity
        similarities = cosine_similarity([input_embedding], embedding_matrix)[0]
        similar_movies = df.iloc[np.argsort(similarities)[::-1][1:top_n+1]]
        return similar_movies[['title', 'genre', 'rating']]
    else:
        # Fallback to fetching recommendations from TMDb or MovieLens if not found in local dataset
        tmdb_recommendations = fetch_tmdb_recommendations(movie_title)
        movielens_recommendations = fetch_movielens_recommendations(movie_title)

        if tmdb_recommendations is not None:
            return tmdb_recommendations
        elif movielens_recommendations is not None:
            return movielens_recommendations
        else:
            return None

# Enhanced regex title extraction as a fallback
def extract_movie_title(user_message):
    # Adjusting the pattern to capture just the title
    pattern = r'(?:love|liked|like|loved|watched|seen|recommend|suggest)\s+([A-Za-z0-9\s\']+)'
    match = re.search(pattern, user_message, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        # If the above pattern doesn't match, try to extract the title using a simpler pattern
        pattern = r'\b([A-Za-z0-9\s\']+)'
        match = re.search(pattern, user_message)
        if match:
            return match.group(0).strip()
    return None

# Test the functions
user_message = input("how can i assist you today ?")
movie_title = extract_movie_title_roberta(user_message)
if movie_title is None:
    movie_title = extract_movie_title(user_message)

if movie_title:
    print(f"Extracted movie title: {movie_title}")
    recommended_movies = recommend_similar_movies(movie_title, df)
    print(f"Recommended movies: {recommended_movies}")
else:
    print("No movie title extracted")
