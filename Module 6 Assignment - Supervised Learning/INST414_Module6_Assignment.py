import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

weather = pd.read_csv("GlobalWeatherRepository.csv")
print(weather.head())

# specify the only features going to be used for supervised learning in a new dataframe
weather_sl = weather.loc[:, ['location_name', 'temperature_fahrenheit', 'condition_text','wind_mph', 'uv_index']]
print(weather_sl.head())
weather_sl["PredictedWeather"] = ""

#apply supervised learning model - classification

weather_sl = weather_sl.dropna()

# Standardize the text before any processing
weather_sl['condition_text'] = weather_sl['condition_text'].str.strip().str.lower()

le = LabelEncoder()
y = le.fit_transform(weather_sl['condition_text'])
X = weather_sl[['temperature_fahrenheit', 'wind_mph', 'uv_index']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
predicted_labels = le.inverse_transform(predictions)

weather_sl.loc[X_test.index, 'PredictedWeather'] = predicted_labels

# check predictions
print("Dataframe with new column containing predictions:\n")
predictions_df = weather_sl.loc[X_test.index].copy()

predictions_df['PredictedWeather'] = predicted_labels

# Display only the locations that were part of the test set
print(predictions_df[['location_name','temperature_fahrenheit', 'wind_mph', 'uv_index',
                      'condition_text', 'PredictedWeather']])
