"""
sudo pip3 install flask
sudo pip3 install pandas
sudo pip3 install scikit-learn
"""

#Required
from flask import Flask, jsonify, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

#Read dataset
car = pd.read_csv('Car-Price-Estimator-API/data/dataset.csv')


#Expose endpoint
@app.route('/api/price', methods=["GET"])
def get_price():
    try:
        car_make = request.args.get("make")
        car_name = request.args.get("name")
        year = request.args.get("year")
        odometer = request.args.get("odometer")

        if int(year) < 2011:
            model = pickle.load(open('Car-Price-Estimator-API/model/LinearRegressionModel-1.pkl', 'rb'))
        elif int(year) >= 2011:
            model = pickle.load(open('Car-Price-Estimator-API/model/LinearRegressionModel-2.pkl', 'rb'))

        prediction = model.predict(pd.DataFrame(columns=['model', 'manufacturer', 'year', 'odometer'],
                                                data=np.array([car_name, car_make, year, odometer]).reshape(1, 4)))
        price = prediction[0]

        if price < 0:
            raise Exception
    except:
        return jsonify(resultvalid=0, message='car not found!', price=0)

    return jsonify(resultvalid=1, message='car found!', price=round(price, 2))


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
