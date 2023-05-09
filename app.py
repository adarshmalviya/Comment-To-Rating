from flask import Flask, request
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from flask_cors import CORS
import pickle

# Load Model
model = load_model('LSTM_Model.h5')

# Load Tokenizer
with open('tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)


def rate(p):
    return (p*5)


app = Flask(__name__)
CORS(app)


@app.route("/")
def index():
    return "Review app working"


@app.route("/postreview", methods=["POST"])
def postreview():
    # Get the data posted from the form
    message = request.get_json(force=True)
    review = message['review']

    # assign the review text to a variable
    a = [review]

    # predict the outcome
    pred = model.predict(pad_sequences(
        tokenizer.texts_to_sequences(a), maxlen=100, padding='post', truncating='post'))
    value = rate(pred.item(0, 0))

    return str(value)[:3]


print("App Running!")

if __name__ == "__main__":
    app.run()
