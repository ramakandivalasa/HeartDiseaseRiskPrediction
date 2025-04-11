from flask import *
from flask_pymongo import PyMongo
app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://localhost:27017/membership"
mongo = PyMongo(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/join', methods=['POST'])
def join():
    age = request.form['age']
    sport = request.form['sport']
    time = request.form['preferredTime']

    mongo.db.members.insert_one({
        "age": int(age),
        "sport": sport,
        "time": time
    })
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)
