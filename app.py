
import numpy as np
from flask import Flask, request, jsonify, render_template

import pickle


app = Flask(__name__)
model = pickle.load(open('house.pkl','rb')) 


@app.route('/')
def home():
  
    return render_template("Index.html")
  
@app.route('/predict',methods=['GET'])
def predict():
    
    
    
    '''
    For rendering results on HTML GUI
    '''
    exp = float(request.args.get('exp'))
    
    prediction = model.predict([[exp]])
    
        
    return render_template('Index.html', prediction_text='Regression Model  has predicted price for given square foot is : {}'.format(prediction))


if __name__ == "__main__":
    app.run(debug=True)
