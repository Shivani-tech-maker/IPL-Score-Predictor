from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the pre-trained model
pipe = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Capturing data typed into the web form by the user
    batting_team = request.form['batting_team']
    bowling_team = request.form['bowling_team']
    city = request.form['city']
    current_score = int(request.form['current_score'])
    overs = float(request.form['overs'])
    wickets = int(request.form['wickets'])
    runs_in_prev_5 = int(request.form['runs_in_prev_5'])

    # Feature Engineering calculations
    balls_left = 120 - (int(overs) * 6 + (overs - int(overs)) * 10)
    wickets_left = 10 - wickets
    crr = current_score / overs if overs > 0 else 0

    # Formatting into a DataFrame exactly as the model expects
    input_df = pd.DataFrame({'batting_team': [batting_team], 'bowling_team': [bowling_team],
                             'city': [city], 'current_score': [current_score],
                             'balls_left': [balls_left], 'wickets_left': [wickets_left],
                             'crr': [crr], 'last_five': [runs_in_prev_5]})

    # Pipeline predicts the score
    result = pipe.predict(input_df)
    predicted_score = int(result[0])

# Reload webpage with the prediction AND the current match stats for the graph
    return render_template('index.html', result=predicted_score, current_score=current_score, overs=overs)
if __name__ == '__main__':
    app.run(debug=True)