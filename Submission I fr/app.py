import gradio as gr
import pandas as pd
import requests
import numpy as np
import pickle
from geopy.distance import geodesic

# Load trained model
with open("trained_apartment_model.pkl", "rb") as f:
    model, scaler = pickle.load(f)

# Load stops dataset
stops_df = pd.read_csv("stops.csv")

# Swiss geolocation API
BASE_URL = "https://api3.geo.admin.ch/rest/services/api/SearchServer?"

def get_geolocation(address):
    params = {"searchText": address, "origins": "address", "type": "locations"}
    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        results = data.get("results", [])
        if results:
            location = results[0]["attrs"]
            return location.get("lat"), location.get("lon")
    return None, None

def get_closest_stop(lat, lon):
    if lat is None or lon is None:
        return "Unknown"
    stops_df["distance"] = stops_df.apply(lambda row: geodesic((lat, lon), (row["stop_lat"], row["stop_lon"])).meters, axis=1)
    return stops_df.loc[stops_df["distance"].idxmin(), "stop_name"]

def predict_price(address, num_rooms, area, luxurious, zurich_city):
    lat, lon = get_geolocation(address)
    closest_stop = get_closest_stop(lat, lon)
    input_data = np.array([[num_rooms, area, int(luxurious), int(zurich_city)]])
    input_data_scaled = scaler.transform(input_data)
    predicted_price = model.predict(input_data_scaled)[0]
    return f"CHF {predicted_price:,.2f}", closest_stop

demo = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Textbox(label="Address"),
        gr.Number(label="Number of rooms"),
        gr.Number(label="Area in m²"),
        gr.Checkbox(label="Luxurious"),
        gr.Checkbox(label="Zurich City"),
    ],
    outputs=[
        gr.Textbox(label="Estimated Price"),
        gr.Textbox(label="Closest Stop")
    ],
    title="Apartment Price Predictor",
    description="Predict the price of an apartment in Zurich based on its location and features."
)

demo.launch()
