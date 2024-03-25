from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('dt.pkl', 'rb'))

app = Flask(__name__)



@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    temp1 = request.form['a']
    temp2 = request.form['b']
    rh1 = request.form['c']
    rh2 = request.form['d']
    rain = request.form['e']
    arr = np.array([[temp1, temp2, rh1, rh2, rain]])
    pred = model.predict(arr)
    inc = pred[0][0]
    sev = pred[0][1]
    stage = request.form.get('stages') 
    return render_template('result.html',sev=sev,inc=inc,stage=stage)
   

if __name__ == "__main__":
    app.run(debug=True)

        