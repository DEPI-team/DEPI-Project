# 🎬 **Filmoria: A Comprehensive Movie Recommendation & Recognition System**

Welcome to **Filmoria**, an innovative mobile application that revolutionizes the movie-watching experience. Whether you're looking for tailored movie recommendations, curious about a short reel you've watched, or need a movie suggestion via chatbot, Filmoria has you covered!

---

## 🚀 **Features**
- **Personalized Movie Recommendations**: Suggests movies based on user preferences such as favorite genres or specific movies.
- **Reel Detection**: Allows users to input reel links and identifies the movie name and the cast members appearing in the reel.
- **Interactive Chatbot**: Chat with our NLP-powered chatbot to receive suggestions like:
  - "I liked *Interstellar*. What should I watch next?"
  - "Recommend action movies!"
- **User-Friendly Interface**: Developed using Flutter for seamless navigation and engaging visuals.

---

## 📚 **How It Works**
1. **Data Collection**: 
   - **IMDbPY**: Extracts movie metadata (title, overview, rating, etc.).
   - **TMDb API**: Provides high-resolution actor images and reel detection support.
   - **MovieLens Dataset**: Includes user preferences, ratings, and tags for collaborative filtering.
   - **IMDb Database**: Fetches reviews, cast, and director details.

2. **Data Preprocessing**:
   - Cleaned and prepared datasets for machine learning models using Python libraries like Pandas, NumPy, and IMDbPY.
   - Enhanced data representation with techniques like tokenization and semantic embeddings.

3. **Recommendation System**:
   - Final model powered by **DistilBERT-base-uncased** for meaningful feature extraction.
   - Models experimented with include cosine similarity, LSTM, bi-directional LSTM, and LLAMA models.

4. **Reel Detection**:
   - Utilizes **OpenCV**, **face_recognition**, and **yt_dlp** for video processing and actor identification.

5. **Deployment**:
   - Backend system developed using **Flask** for real-time recommendations and chatbot responses.
   - Frontend built with **Flutter** for cross-platform compatibility.

---

## 🛠️ **Tools & Libraries**
- **Programming Languages**: Python, Dart
- **Machine Learning**: TensorFlow, PyTorch, Hugging Face Transformers
- **APIs**: IMDbPY, TMDb API
- **Data Processing**: Pandas, NumPy
- **Backend**: Flask
- **Frontend**: Flutter
- **Other Libraries**: OpenCV, face_recognition, yt_dlp, MLflow

---

## 🎯 **Project Goals**
- Simplify the movie discovery process with accurate and personalized recommendations.
- Enhance user interaction by integrating advanced NLP models for movie-related queries.
- Provide a one-stop solution for reel detection and actor recognition.

---

## 🔮 **Future Plans**
- **Backend Development**: Implementing a robust backend system to handle complex queries and data processing more efficiently.
- **Feature Extraction Model**: Developing a custom model for more accurate feature extraction and recommendations.
- **Expanding Dataset**: Increasing the size and diversity of our dataset for better model generalization.
- **Advanced Models**: Experimenting with state-of-the-art models to improve recommendations, such as transformers and GANs.
- **Application Deployment**: Hosting the application on an online server for real-world usage.
- **User Experience**: Adding features like dark mode, personalized themes, and multilingual support.
- **Real-time Data**: Incorporating trending movies and real-time recommendations.

---

## 📱 **Screenshots**



https://github.com/user-attachments/assets/a07084d7-f19c-4cd5-97de-c972ca5c4f59



https://github.com/user-attachments/assets/b93fe18c-9069-4f3d-b324-297b330803ae





---

## 💡 **Get Started**

1. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/filmoria.git

2. Navigate to the project directory:  
   ```bash
   cd filmoria

3.Install the required dependencies:

    
     pip install -r requirements.txt

4. Run the application backend:

    ```bash
    python app.py

5. Launch the frontend using Flutter:

    ```bash
    flutter run

## 📊 Results 
The DistilBERT-based recommendation model provided higher accuracy and relevance compared to traditional models like TF-IDF and LSTM.
Successful integration of movie reel detection to identify films and cast members from short clips.
User feedback showed improved recommendation relevance after incorporating advanced models and expanding the dataset.


## 🤝 Contributors 
Team Members:

George Esbergen Sedky Reyad

Hassan Mohammed Basuony Nagy

Abdulrahman Hisham Kamel Mahmoud

Omar Khilad Shepl Emara


##  📬 Contact 
For questions or feedback, please reach out to us via:

GitHub Issues

Email : Abdulrahmanhishamk@gmail.com

We hope you enjoy Filmoria and find your next favorite movie! 🎬🍿
