import pickle
from flask import Flask,request,render_template,jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application

linear_model = pickle.load(open('models/linear.pkl','rb'))
standard_scaler = pickle.load(open('models/scaler.pkl','rb'))

@app.route('/',methods = ['GET','POST'])
def index():
    if request.method=="POST":
        Age=float(request.form.get('Age'))
        Experience = float(request.form.get('Experience'))

        new_data = standard_scaler.transform([[Age,Experience]])
        results = linear_model.predict(new_data)

        results_rounded = round(results[0])

        if Age >= 15:
            prediction = {results_rounded}
        else:
            prediction = "The Age is not correct to fetch the details"


        return render_template('index.html',results=prediction)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
