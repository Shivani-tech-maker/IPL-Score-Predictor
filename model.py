import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error

df = pd.read_csv('cleaned_ipl_data.csv') 

X = df.drop(columns=['total'])
y = df['total']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# We removed drop='first' and added handle_unknown='ignore'
trf = ColumnTransformer([
    ('trf', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), ['batting_team', 'bowling_team', 'city'])
], remainder='passthrough')

# Creating a Pipeline. It ensures incoming web data is encoded first, 
# then passed to the Random Forest model automatically.
pipe = Pipeline(steps=[
    ('step1', trf),
    ('step2', RandomForestRegressor(n_estimators=100, random_state=42))
])

# fit() is where the model looks at historical data and learns patterns.
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

print(f"R2 Score : {r2_score(y_test, y_pred)}")
print(f"MAE : {mean_absolute_error(y_test, y_pred)}")

# Saving the trained model. We use pickle so we don't retrain on every website visit.
pickle.dump(pipe, open('model.pkl', 'wb'))
print("Model trained and saved successfully!")