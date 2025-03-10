import gradio as gr
import pandas as pd
import requests
import pickle
from sklearn.preprocessing import LabelEncoder

# Load trained model
model_filename = "Submission I\model.pkl"
with open(model_filename, mode="rb") as f:
    model, scaler, categorical_cols = pickle.load(f)

# Function to get geolocation data
def get_geolocation(address):
    BASE_URL = "https://api3.geo.admin.ch/rest/services/api/SearchServer?"
    params = {"searchText": address, "origins": "address", "type": "locations"}
    response = requests.get(BASE_URL, params=params)
    
    if response.status_code == 200:
        data = response.json()
        results = data.get("results", [])
        if results:
            location = results[0]["attrs"]
            return location.get("lat"), location.get("lon"), location.get("x"), location.get("y")
    
    return None, None, None, None  # Default values if no result is found

# Prediction function
def predict_price(address, rooms, area, luxurious, zurich_city):
    # Get geolocation data
    lat, lon, x, y = get_geolocation(address)
    
    # Convert input features into a DataFrame
    input_data = pd.DataFrame([[rooms, area, luxurious, zurich_city, lat, lon, x, y]],
                              columns=['Number of rooms', 'Area in m²', 'Luxurious', 'Zurich City', 'lat', 'lon', 'x', 'y'])
    
    # Encode categorical variables
    for col in categorical_cols:
        le = LabelEncoder()
        if col in input_data.columns:
            input_data[col] = le.fit_transform(input_data[col])
    
    # Scale numerical features
    input_data = scaler.transform(input_data)
    
    # Predict price
    prediction = model.predict(input_data)[0]
    return f"Predicted Price: {prediction}, Location: ({lat}, {lon})"

# Create Gradio interface
demo = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Textbox(label="Address"),
        gr.Number(label="Number of rooms"),
        gr.Number(label="Area in m²"),
        gr.Checkbox(label="Luxurious"),
        gr.Checkbox(label="Zurich City"),
    ],
    outputs="text",
    examples=[
        ["Zürcherstrasse 1, 8173 Neerach", 3.5, 65, False, False],
        ["Badenerstrasse 123, 8004 Zürich", 4, 98, False, True],
        ["Robert-Stephenson-Weg 47, 8004 Zürich", 4.5, 148, False, True]
    ],
    title="Apartment Price Predictor",
    description="Predict the price of an apartment in Zurich based on its location and features."
)

demo.launch()
