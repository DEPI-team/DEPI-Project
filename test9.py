import cv2
import face_recognition
import numpy as np
import logging
import os
import yt_dlp as youtube_dl
import requests
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])

# TMDB API key
API_KEY = '2c4db75d543cd80b7bef3eba708d4f79'  # Replace with your API Key

def download_youtube_video(url, output_path):
    """Download a YouTube video using yt-dlp."""
    try:
        ydl_opts = {
            'format': 'mp4',
            'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
            'noplaylist': True,
            'quiet': False
        }
        
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            video_title = info_dict.get('title', 'video')
            video_filename = ydl.prepare_filename(info_dict)
            logging.info(f"Downloaded video: {video_title}")
            return video_filename
    except Exception as e:
        logging.error(f"Error downloading video: {e}")
        return None

def load_known_faces(image_folder):
    """Load known faces from images in a folder."""
    known_face_encodings = []
    known_face_names = []
    cache_file = os.path.join(image_folder, 'face_encodings.pkl')

    # Load from cache if it exists
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            known_face_encodings, known_face_names = pickle.load(f)
            logging.info(f"Loaded {len(known_face_encodings)} faces from cache.")
            return known_face_encodings, known_face_names

    # If not in cache, load faces
    for filename in os.listdir(image_folder):
        image_path = os.path.join(image_folder, filename)
        
        if not (filename.lower().endswith('.jpg') or filename.lower().endswith('.png')):
            continue

        try:
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image)
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            if not face_encodings:
                logging.warning(f"No faces found in image {image_path}.")
                continue
            
            for face_encoding in face_encodings:
                known_face_encodings.append(face_encoding)
                known_face_names.append(os.path.splitext(filename)[0])

        except Exception as e:
            logging.error(f"Error loading image {image_path}: {e}")

    logging.info(f"Loaded {len(known_face_encodings)} known faces.")
    
    # Save data to cache
    with open(cache_file, 'wb') as f:
        pickle.dump((known_face_encodings, known_face_names), f)

    return known_face_encodings, known_face_names

def process_frame(frame, known_face_encodings, known_face_names):
    """Process a video frame to detect known faces."""
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    face_names = []
    face_count = {}

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances_current = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances_current)
        name = "Unknown"
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        
        if name != "Unknown":
            if name not in face_count:
                face_count[name] = 0
            face_count[name] += 1

    return face_locations, face_count

def get_actor_movies_tmdb(api_key, actor_name):
    """Get movies featuring the actor using TMDB."""
    try:
        search_url = f"https://api.themoviedb.org/3/search/person?api_key={api_key}&query={actor_name}"
        response = requests.get(search_url)

        if response.status_code != 200:
            logging.error(f"Error fetching actor data: {response.status_code} - {response.text}")
            return []

        data = response.json()

        if not data.get('results'):
            logging.info(f"No actor page found for {actor_name}")
            return []

        actor_id = data['results'][0]['id']

        movie_url = f"https://api.themoviedb.org/3/person/{actor_id}/movie_credits?api_key={api_key}"
        response = requests.get(movie_url)

        if response.status_code != 200:
            logging.error(f"Error fetching movie credits: {response.status_code} - {response.text}")
            return []

        movie_data = response.json()
        movies = [movie['title'] for movie in movie_data.get('cast', [])]

        return movies
    except Exception as e:
        logging.error(f"Error retrieving movies: {e}")
        return []

def get_common_movies(movies_dict):
    """Find common movies among actors."""
    if not movies_dict:
        return []

    common_movies = set(movies_dict[list(movies_dict.keys())[0]])
    for movies in movies_dict.values():
        common_movies.intersection_update(movies)
    
    return list(common_movies)

def analyze_video(video_path, known_face_encodings, known_face_names):
    """Analyze the video to detect known faces."""
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        logging.error(f"Cannot open video file {video_path}.")
        return
    
    logging.info("Successfully opened the video file.")
    frame_count = 0
    total_face_count = {}

    while True:
        ret, frame = video_capture.read()
        if not ret:
            logging.info("Video ended or no more frames found.")
            break
        
        face_locations, face_count = process_frame(frame, known_face_encodings, known_face_names)
        
        for name in face_count:
            if name not in total_face_count:
                total_face_count[name] = 0
            total_face_count[name] += face_count[name]

        frame_count += 1

    video_capture.release()
    cv2.destroyAllWindows()

    logging.info(f"Processed {frame_count} frames.")

    # Get top actors
    top_actors = get_top_actors(total_face_count)
    
    # Get movies for each of the top actors
    movie_appearances = {}
    for actor in top_actors:
        movie_appearances[actor] = get_actor_movies_tmdb(API_KEY, actor)
    
    # Print only common movies among the top actors
    common_movies = get_common_movies(movie_appearances)
    if common_movies:
        print(f"Common movies among actors: {', '.join(common_movies)}")
    else:
        print("No common movies among actors.")

def get_top_actors(face_count, min_appearance=10, top_n=2):
    """Get top actors appearing in the video."""
    top_actors = {actor: count for actor, count in face_count.items() if count >= min_appearance}
    top_actors_sorted = sorted(top_actors.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [actor for actor, _ in top_actors_sorted]

def test_video_read(video_path):
    """Test video reading to ensure it works correctly."""
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        logging.error(f"Cannot open video file {video_path}.")
        return
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            logging.info("Video ended or no more frames found.")
            break
        
        cv2.imshow('Video Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logging.info("Video stopped by user.")
            break
    
    video_capture.release()
    cv2.destroyAllWindows()

def main():
    youtube_url = 'https://youtu.be/7Zw-qPlydTI?si=DqhTsZGpq42z48O8'
    download_folder = 'D:/project/downloaded_videos'
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    video_path = download_youtube_video(youtube_url, download_folder)
    if not video_path:
        return

    image_folder = 'D:/project/faces4'
    
    known_face_encodings, known_face_names = load_known_faces(image_folder)
    
    # Test video reading
    test_video_read(video_path)
    
    analyze_video(video_path, known_face_encodings, known_face_names)

if __name__ == "__main__":
    main()
