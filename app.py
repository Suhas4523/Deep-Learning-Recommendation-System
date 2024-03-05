from flask import Flask, render_template, request
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import numpy as np

app = Flask(__name__)

# Load your trained model
model = load_model('my_model.h5')

# Load necessary data
ratings = pd.read_csv('Recommender-Systems-with-Collaborative-Filtering-and-Deep-Learning-Techniques\\ml-latest-small\\ratings.csv')
movies = pd.read_csv('Recommender-Systems-with-Collaborative-Filtering-and-Deep-Learning-Techniques\\ml-latest-small\\movies.csv')  # Update with the correct path

# Create a dictionary to map movie names to movie IDs
movie_name_to_id = dict(zip(movies['title'].str.lower(), movies['movieId']))

# Adjust the path accordingly
label_encoder_user = LabelEncoder()
label_encoder_movie = LabelEncoder()
ratings['userId'] = label_encoder_user.fit_transform(ratings['userId'])
ratings['movieId'] = label_encoder_movie.fit_transform(ratings['movieId'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_id = int(request.form['user_id'])
        input_movie_name = request.form['movie_name']

        # Encode user ID
        encoded_user_id = label_encoder_user.transform([user_id])[0]

        # Perform a case-insensitive search for the movie name
        matched_movie_names = [movie_name for movie_name in movie_name_to_id.keys() if input_movie_name.lower() in movie_name.lower()]

        if matched_movie_names:
            # Use the first matched movie name
            matched_movie_name = matched_movie_names[0]

            # Get movie ID from the dictionary
            movie_id = movie_name_to_id[matched_movie_name]

            # Encode movie ID
            encoded_movie_id = label_encoder_movie.transform([movie_id])[0]

            # Make sure to convert the IDs to NumPy arrays
            encoded_user_id = np.array([encoded_user_id])
            encoded_movie_id = np.array([encoded_movie_id])

            # Make a prediction using the model
            prediction = model.predict([encoded_user_id, encoded_movie_id])

            # Map the predicted rating to the predefined scale
            if prediction <= 1:
                predicted_label = "Strongly Dislike"
            elif 1 < prediction <= 2:
                predicted_label = "Mostly Dislike"
            elif 2 < prediction <= 3:
                predicted_label = "Neutral"
            elif 3 < prediction <= 4:
                predicted_label = "Mostly Like"
            else:
                predicted_label = "Strongly Like"

            return render_template('result.html', prediction=prediction, label=predicted_label, movie_name=matched_movie_name)

        else:
            return render_template('error.html', message='Movie not found in the dataset')

if __name__ == '__main__':
    app.run(debug=True)
