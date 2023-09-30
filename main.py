from flask import Flask, render_template, request
import pickle
import iris

main = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@main.route('/')
def home():
    return render_template('index.html')


@main.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        sepal_length = float(request.form['sepal_length'])
        petal_length = float(request.form['petal_length'])
        trained_model = iris.training_model()
        prediction_value = trained_model.predict([[sepal_length, petal_length]])

        setosa = 'setosa'
        versicolor = 'versicolor'
        virginica = 'verginica'

        if prediction_value == 0:
            return render_template('index.html', setosa=setosa)
        elif prediction_value == 1:
            return render_template('index.html', versicolor=versicolor)
        else:
            return render_template('index.html', virginica=virginica)

    return render_template('index.html')


if __name__ == '__main__':
    main.run(debug=True)

