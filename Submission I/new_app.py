import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import gradio as gr

# Load dataset
df = pd.read_csv('apartments_data_enriched_with_new_features.csv')

# Preprocess data (assuming relevant columns exist)
features = ['Number_of_rooms', 'Area', 'Luxurious', 'Zurich_City']  # Adjust column names as needed
target = 'Price'  # Adjust target column name if necessary

# Encode categorical features if needed
df['Luxurious'] = df['Luxurious'].astype(int)
df['Zurich_City'] = df['Zurich_City'].astype(int)

# Splitting data
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Save the trained model
with open("trained_apartment_model.pkl", "wb") as f:
    pickle.dump((model, scaler), f)

# Function for Gradio prediction
def predict_price(address, num_rooms, area, luxurious, zurich_city):
    with open("trained_apartment_model.pkl", "rb") as f:
        model, scaler = pickle.load(f)
    
    input_data = np.array([[num_rooms, area, int(luxurious), int(zurich_city)]])
    input_data_scaled = scaler.transform(input_data)
    predicted_price = model.predict(input_data_scaled)[0]
    return f"Estimated price for {address}: CHF {predicted_price:,.2f}"

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
