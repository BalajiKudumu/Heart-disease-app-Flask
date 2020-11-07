import numpy as np
from flask import Flask,request,jsonify,render_template

import pickle

app=Flask(__name__)
model=pickle.load(open('heartdisease.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features=[int(float(x)) for x in request.form.values()]
    final_features=[np.array(int_features)]
    prediction=model.predict(final_features)

    output=prediction[0]
    if output==0:

        return render_template('index.html', prediction_text="The person is suffering from Heart Disease, recommended for further diagnosis ")
    else:
        return render_template('index.html', prediction_text="The person is not suffering from Heart Disease")


@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.get_json(force=True)
    prediction=model.predict([np.array(list(data.values()))])
    output=prediction[0]
    return jsonify(output)


if __name__=="__main__":
    app.run(debug=True)