from flask import Flask

app = Flask(__name__)

@app.route('/api/price/<string:car_make>/<string:car_name>/<int:year>/<int:odometer>', methods=["GET"])
def get_price(car_make, car_name, year, odometer):
    return '<h1>Welcome to Flask! {0} {1} {2} {3} </h1>'.format(car_make, car_name, year, odometer)

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
