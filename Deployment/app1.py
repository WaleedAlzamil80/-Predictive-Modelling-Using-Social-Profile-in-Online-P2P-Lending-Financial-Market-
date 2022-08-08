import numpy as np
import pickle
from flask import Flask, render_template, request, jsonify, Markup

app = Flask(__name__)
model=pickle.load(open('LogisticRegressionCompletedFINAL.pkl','rb'))

@app.route('/')
def Home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    interestRate = float_features[-1]
    float_features.pop()
    features=[np.array(float_features)]
    predictions=model.predict(features)

    ## Calculate ROI
    currInvestement = float_features[5] * (1 + interestRate) ** (float_features[10] / 12)

    costOfInvestment1 = float_features[5]
    ROI1 = (currInvestement - costOfInvestment1) / costOfInvestment1
    
    ## Calculate ROI when fees taken into account
    costOfInvestment2 = float_features[5] * ((float_features[6]) ** (float_features[10] / 12)) + float_features[5]
    ROI2 = (currInvestement - costOfInvestment2) / costOfInvestment2

    if predictions[0] == 1:
        return render_template('index.html', prediction_text=Markup(f"The Loan Should Be Paid <br/> ROI (Return on Investment): {ROI1:.2f}% <br/> ROI (Return on Investment when fees are taken into account): {ROI2:.2f}%"))
    else:
        return render_template('index.html', prediction_text=Markup(f"The Loan Should Be NOT Paid <br/> ROI (Return on Investment): {ROI1:.2f}% <br/> ROI (Return on Investment when fees are taken into account): {ROI2:.2f}%"))

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data=request.get_json(force=True)
    predictions=model.predict([np.array(list(data.values()))])

    return jsonify(predictions[0])


if __name__ == "__main__":
    app.run(debug=True)

#if __name__ == "__main__":
 #   from waitress import serve
  #  serve(app, host="0.0.0.0", port=8080)
  