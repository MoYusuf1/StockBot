import requests
import numpy as np
from sklearn.linear_model import LinearRegression

url = "https://twelve-data1.p.rapidapi.com/stocks"

querystring = {"exchange": "NASDAQ", "format": "json"}

data = requests.get(url)

headers = {
    "X-RapidAPI-Key": "d3590f15a6mshcfacd3021e36ddfp1f00c5jsn3670f97cdc18",
    "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
}

response = requests.request("GET", url, headers=headers)

print(response.text)

# Convert the data to a numpy array
data_array = np.array(data.json())
X = data_array[:, 0]
y = data_array[:, 1]

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Evaluate model
score = model.score(X, y)
print(score)

# This code will retrieve the data from the API, convert it to a numpy array, and then train and evaluate a linear
# regression model on the data.
