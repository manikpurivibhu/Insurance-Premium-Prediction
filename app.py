from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import requests
import pickle

app = Flask(__name__)

#Load Machine Learning Model
with open('insurance_premium_prediction_model.pkl', 'rb') as file:  
    model = pickle.load(file)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        return render_template('home.html')
    
    return render_template('home.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            age      = int(request.form['age'])
            age      = (age-39)/(51-27)                  #Robust Scaling 
            
            bmi      = float(request.form['bmi'])        #OneHotEncoding
            if bmi in np.arange(0, 18.5, 0.1):
                bmi = [0,0,0,1]
            elif bmi in np.arange(18.5, 24.9, 0.1):
                bmi = [1,0,0,0]
            elif bmi in np.arange(24.9, 29.9, 0.1):
                bmi = [0,0,1,0]
            else:
                bmi = [0,1,0,0]

            children = int(request.form['children'])     #Robust Scaling
            children = (children-1)/(2-0)               

            smoker   = request.form['smoker']            #LabelEncoding
            if smoker == 'yes':
                smoker = 1
            else:
                smoker = 0

            region   = request.form['region']            #OneHotEncoding
            if region == 'northeast':
                regional_area = [1, 0, 0, 0]
            elif region == 'northwest':
                regional_area = [0, 1, 0, 0]
            elif region == 'southeast':
                regional_area = [0, 0, 1, 0]
            else:
                regional_area = [0, 0, 0, 1]
    
            sex  = request.form['sex']                   #OneHotEncoding
            if sex == 'male':
                sex=0
            else:
                sex=1

            
            pred_args = [age, sex]+ bmi + [children, smoker] + regional_area
            pred_args_arr = np.array(pred_args)
            pred_args_arr = pred_args_arr.reshape(1, -1)
            model_prediction = (model.predict(pred_args_arr) * 11899.63) + 9382.03        #Unpack Robust Scaled predicted value
            model_prediction = round(float(model_prediction), 2)
            

        except (ValueError, UnboundLocalError) as e:
             return render_template('incorrectinput.html')

    return render_template('predict.html', prediction = model_prediction)




if __name__ == '__main__':
    app.run(port=8000)