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

## 🌟 **Future Plans**
- Expand the dataset with more movie sources and user ratings.
- Create a proprietary feature extraction model to improve recommendation accuracy.
- Deploy the app on an online server for global accessibility.
- Incorporate real-time trending movies into recommendations.
- Integrate multi-language support for a wider audience.

---

## 📱 **Screenshots**



https://github.com/user-attachments/assets/ede38667-5b29-4b9a-b955-2df5984f8182

https://github.com/user-attachments/assets/1e8bde50-4919-42e8-931e-3b2091bfe489



---

## 💡 **Get Started**
1. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/filmoria.git







---

## 🔮 **Future Plans**
We aim to continually improve and expand Filmoria by:
- **Backend Development**: Implementing a robust backend system to handle complex queries and data processing more efficiently.
- **Feature Extraction Model**: Developing a custom model for more accurate feature extraction and recommendations.
- **Expanding Dataset**: Increasing the size and diversity of our dataset for better model generalization.
- **Advanced Models**: Experimenting with state-of-the-art models to improve recommendations, such as transformers and GANs.
- **Application Deployment**: Hosting the application on an online server for real-world usage.
- **User Experience**: Adding features like dark mode, personalized themes, and multilingual support.

---

## 🧠 **Key Features**
1. **Movie Recommendations**:
   - Suggests movies based on user preferences and similarities using advanced NLP models like DistilBERT.
   - Chatbot interaction for personalized queries such as "I loved Interstellar. What should I watch next?"

2. **Reel Detection**:
   - Detects movies and actors in uploaded reels using actor profiles and facial recognition.

3. **Collaborative Filtering**:
   - Uses MovieLens data for user-specific recommendations based on preferences and ratings.

4. **Seamless Mobile Application**:
   - Intuitive Flutter-based interface for users to interact with the system effortlessly.

---

## 🚀 **Technologies Used**
### Machine Learning:
- **NLP Models**: DistilBERT-base-uncased for movie similarity embedding.
- **Collaborative Filtering**: User-based and item-based filtering from MovieLens data.
- **Feature Extraction**: Advanced embeddings for text data preprocessing.

### Tools & Frameworks:
- Python: Pandas, NumPy, Scikit-learn, TensorFlow, PyTorch.
- IMDbPY & TMDb API: For data collection and enrichment.
- Flask: Backend server handling API requests.
- Flutter: Frontend mobile app development.
- MySQL: Database for storing and managing movie metadata.

---

## 📊 **Results**
- The **DistilBERT-based recommendation model** provided higher accuracy and relevance compared to traditional models like TF-IDF and LSTM.
- Successful integration of **movie reel detection** to identify films and cast members from short clips.
- User feedback showed **improved recommendation relevance** after incorporating advanced models and expanding the dataset.

---

## 📜 **License**
This project is licensed under the MIT License. See the LICENSE file for details.

---

## 🤝 **Contributors**
### Team Members:
- George Esbergen Sedky Reyad
- Hassan Mohammed Basuony Nagy
- Abdulrahman Hisham Kamel Mahmoud
- Omar Khilad Shepl Emara

---

## 📬 **Contact**
For questions or feedback, please reach out to us via:
- [GitHub Issues](https://github.com/yourusername/filmoria/issues)
- [Email](mailto:abdulrahmanhishamk@gmail.com)

---

**We hope you enjoy Filmoria and find your next favorite movie!** 🎬🍿

