'''
this file is for backend
'''
import flask
import numpy as np
from flask import Flask,render_template,request
import pickle
import sklearn
from sklearn.linear_model import LinearRegression
with open("SLR_MODEL.pkl","rb") as f:
    m = pickle.load(f) # m -> model


app = Flask(__name__)

@app.route('/')
def check():
    return render_template("index.html")

@app.route("/predict",methods = ['GET','POST'])
def fun3():
    a = [float(i) for i in request.form.values()] # 11
    b = [np.array(a)] # [[11]]
    sol = m.predict(b)[0]  # a = [12] -> a[0] -> 12
    return render_template("index.html" , prediction_text = sol)

if __name__ == "__main__":
    app.run(debug=True)