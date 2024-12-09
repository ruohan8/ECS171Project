from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load your pre-trained machine learning model here
with open('model.pkl', 'rb') as f:  # Change the path based on your model
    model = pickle.load(f)
    
with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

with open('pca.pkl', 'rb') as f:
    pca = pickle.load(f)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Get user input from the form
        age = request.form.get("age")
        gender = request.form.get("gender")
        education = request.form.get("education")
        job_category = request.form.get("jobCategory")
        experience = request.form.get("experience")
        country = request.form.get("country")
        race = request.form.get("race")
        senior = request.form.get("senior")
        
        new_data = np.array([age, gender, education, job_category, experience, country, race, senior]).reshape(1, -1)

        # Preprocess the input data for your model (if needed)
        data = pd.DataFrame(new_data, columns=['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience', 'Country', 'Race', 'Senior'])
        categorical_columns = ['Gender', 'Job Title', 'Country', 'Race']
        one_hot_encoded = encoder.transform(data[categorical_columns])
        one_hot_data = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))
        data_encoded = pd.concat([data, one_hot_data], axis=1)
        data_encoded  = data_encoded.drop(categorical_columns, axis=1)
        
        input_data = pca.transform(data_encoded)
        
        # Make prediction using your model
        prediction = model.predict(input_data)  # Assuming a list input

        # Format the prediction for display
        predicted_class = prediction[0]  # Assuming single class output

        return render_template("results.html", prediction=predicted_class)

    else:
        return "Something went wrong. Please try again."

if __name__ == "__main__":
    app.run(debug=True)