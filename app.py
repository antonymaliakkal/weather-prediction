from flask import Flask,request,render_template
import numpy as np
import pickle

app = Flask(__name__)

knn = pickle.load(open('models/knn.pkl','rb'))
rf = pickle.load(open('models/random_forest.pkl','rb'))
ann = pickle.load(open('models/ann.pkl','rb'))
gbc = pickle.load(open('models/gbc.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    a = request.form['Precipitation']
    a = float(a)
    b = request.form['max_temp']
    b = float(b)
    c = request.form['min_temp']
    c = float(c)
    d = request.form['wind']
    d = float(d)
    int_features = [a,b,c,d]
    features = [np.array(int_features)]

    option = request.form.get('option')
    if option == 'knn':
        prediction = knn.predict(features)
    elif option == 'rf':
        prediction = rf.predict(features)
    elif option == 'ann':
        prediction = ann.predict(features)
    elif option == 'gbc':
        prediction = gbc.predict(features)

    print('Predicted' , prediction)

    if prediction == [0]:
        return render_template('index.html', prediction_text='Weather is {}'.format('drizzle'))
    elif prediction == [1]:
        return render_template('index.html', prediction_text='Weather is {}'.format('fog'))
    elif prediction == [2]:
        return render_template('index.html', prediction_text='Weather is {}'.format('rain'))
    elif prediction == [3]:
        return render_template('index.html', prediction_text='Weather is {}'.format('snow'))
    else:
        return render_template('index.html', prediction_text='Weather is {}'.format('sun'))


if __name__ == "__main__":
    app.run(debug = True)  