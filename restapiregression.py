import flask
from flask import Flask, request
import pickle
import pandas as pd

model_linear=pickle.load(open('modelRegression.pkl','rb'))
app=Flask(__name__)
@app.route('/',methods=['GET', 'POST'])
def main():
    return "Flask API is initialised"

@app.route('/predict',methods=['GET'])
def predict():
    if flask.request.method == 'GET':
        age=request.args.get('age')
        sex=request.args.get('sex')
        bmi = request.args.get('bmi')
        children = request.args.get('children')
        smoker = request.args.get('smoker')
        region = request.args.get('region')
        columns=[(age,sex,bmi,children,smoker,region)]
        labels=['age','sex','bmi','children','smoker','region']
        predict=pd.DataFrame.from_records(columns, columns=labels)
        
        predict.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)
        predict.replace({'smoker': {'yes': 0, 'no': 1}}, inplace=True)
        predict.replace({'region': {'southwest': 0, 'northwest': 1, 'southeast': 2, 'northeast': 3}}, inplace=True)
        age = predict.at[0, 'age']
        sex = predict.at[0, 'sex']
        bmi = predict.at[0, 'bmi']
        children = predict.at[0, 'children']
        smoker = predict.at[0, 'smoker']
        region = predict.at[0, 'region']
        # print(age,gender,bmi,children,smoker,region)
        prediction = model_linear.predict([[age, sex, bmi, children, smoker, region]])
        # print(prediction)
        result = 'We think that is {}.'.format(prediction)
        return result
    else:
        print("Wrong Method is selected")

if __name__=="__main__":
    app.run()