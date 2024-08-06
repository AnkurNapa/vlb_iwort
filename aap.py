import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
 
# Load or prepare your dataset
data = {
    'Recipe Name': ['IPA', 'Stout', 'Lager', 'Porter', 'Pale Ale'],
    'Yeast Strain': ['US-05', 'WLP001', 'Safale S-04', 'Wyeast 1056', 'WLP002'],
    'Mash Temp': [67, 68, 66, 65, 69],
    'Boil Time': [60, 75, 60, 90, 60],
    'OG': [1.050, 1.060, 1.040, 1.070, 1.055],
    'FG': [1.010, 1.015, 1.008, 1.020, 1.012],
    'IBU': [40, 50, 20, 60, 35],
    'SRM': [10, 40, 5, 30, 12],
    'ABV': [5.3, 6.5, 4.2, 7.0, 5.5],
    'Sensory Scores': [8, 9, 7, 8.5, 8]
}
df = pd.DataFrame(data)
 
# Encode categorical variables
label_encoders = {}
for column in ['Recipe Name', 'Yeast Strain']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le
 
# Select relevant features
features = ['Recipe Name', 'Yeast Strain', 'Mash Temp', 'Boil Time', 'OG', 'FG', 'IBU', 'SRM', 'ABV']
target = 'Sensory Scores'
numerical_features = ['Mash Temp', 'Boil Time', 'OG', 'FG', 'IBU', 'SRM', 'ABV']
 
# Extract features and target
X = df[features]
y = df[target]
 
# Normalize numerical features
scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])
 
# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)
 
# Streamlit app
st.title("Beer Recipe Optimizer")
 
# User inputs
recipe_name = st.selectbox('Recipe Name', df['Recipe Name'].unique())
yeast_strain = st.selectbox('Yeast Strain', df['Yeast Strain'].unique())
mash_temp = st.slider('Mash Temperature (°C)', 60, 70, 67)
boil_time = st.slider('Boil Time (minutes)', 30, 120, 60)
og = st.number_input('Original Gravity (OG)', 1.030, 1.080, 1.050)
fg = st.number_input('Final Gravity (FG)', 1.000, 1.030, 1.010)
ibu = st.slider('International Bitterness Units (IBU)', 0, 100, 40)
srm = st.slider('Standard Reference Method (SRM)', 0, 50, 10)
abv = st.slider('Alcohol by Volume (ABV)', 0.0, 10.0, 5.0)
 
# Encode user inputs
def encode_input(label_encoders, column, value):
    le = label_encoders[column]
    if value not in le.classes_:
        # Add the unseen label to the classes
        le.classes_ = np.append(le.classes_, value)
    return le.transform([value])[0]
 
user_input = pd.DataFrame([[recipe_name, yeast_strain, mash_temp, boil_time, og, fg, ibu, srm, abv]],
                          columns=features)
 
for column in ['Recipe Name', 'Yeast Strain']:
    user_input[column] = encode_input(label_encoders, column, user_input[column].iloc[0])
 
# Normalize user inputs
user_input[numerical_features] = scaler.transform(user_input[numerical_features])
 
# Predict original rating
original_rating = model.predict(user_input)[0]
 
# Suggest modifications using a simple rule-based system for demonstration
def suggest_modifications(recipe):
    suggested_recipe = recipe.copy()
    suggested_recipe['Mash Temp'] += 0.1
    suggested_recipe['IBU'] += 2
    suggested_recipe['ABV'] += 0.1
    return suggested_recipe
 
suggested_input = suggest_modifications(user_input.copy())
 
# Decode and reverse normalization for display
user_input_decoded = user_input.copy()
suggested_input_decoded = suggested_input.copy()
 
for column in ['Recipe Name', 'Yeast Strain']:
    user_input_decoded[column] = label_encoders[column].inverse_transform([user_input[column].iloc[0]])[0]
    suggested_input_decoded[column] = label_encoders[column].inverse_transform([suggested_input[column].iloc[0]])[0]
 
user_input_decoded[numerical_features] = scaler.inverse_transform(user_input[numerical_features])
suggested_input_decoded[numerical_features] = scaler.inverse_transform(suggested_input[numerical_features])
 
# Predict rating for suggested recipe
suggested_rating = model.predict(suggested_input)[0]
 
# Display results
st.subheader("Original Recipe")
st.write(user_input_decoded)
st.write(f"Predicted Rating: {original_rating:.2f}")
 
st.subheader("Suggested Recipe")
st.write(suggested_input_decoded)
st.write(f"Predicted Rating: {suggested_rating:.2f}")
 
if suggested_rating > original_rating:
    st.success(f"The suggested recipe has a higher predicted rating of {suggested_rating:.2f} compared to the original rating of {original_rating:.2f}.")
else:
    st.warning(f"The suggested recipe has a lower predicted rating of {suggested_rating:.2f} compared to the original rating of {original_rating:.2f}.")