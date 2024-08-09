from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from source.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            Gender=request.form.get('Gender'),
            Married=request.form.get('Married'),
            Dependents=int(request.form.get('Dependents')),
            Education=request.form.get('Education'),
            Self_Employed=request.form.get('Self_Employed'),
            ApplicantIncome=int(request.form.get('ApplicantIncome')),
            CoapplicantIncome=int(request.form.get('CoapplicantIncome')),
            LoanAmount=int(request.form.get('LoanAmount')),
            Loan_Amount_Term=int(request.form.get('Loan_Amount_Term')),
            Credit_History=int(request.form.get('Credit_History')),
            Property_Area=request.form.get('Property_Area')

        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")

        results=predict_pipeline.predict(pred_df)
        print("after Prediction")

        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.run(debug=True)