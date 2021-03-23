from flask import Flask,request,jsonify,render_template
import numpy as np
from model import model

import pickle
app = Flask(__name__)  # creating the Flask class object
mod=pickle.load(open('C:/Users/aditya kumar/PycharmProjects/flask-ML-project/model.pkl', 'rb'))

# def ValuePredictor(to_predict):
#     #to_predict = np.array(to_predict_list).reshape(1, 12)
#     model = pickle.load(open("model.pkl", "rb"))
#     result = model.predict(to_predict)
#     return result


@app.route('/')  # decorator drfines the
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_value():
    if request.method == 'POST':
        features=request.form['TEXT']
        s=mod.predict([features])
        prediction=model.train.target_names[s[0]]
        return render_template('index.html', features=features,prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)