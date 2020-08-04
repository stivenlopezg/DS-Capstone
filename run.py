from flask_restful import Api
from predict import load_model, predict
from flask import Flask, request, render_template


app = Flask(__name__, template_folder="templates")
api = Api(app)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/churn', methods=['GET', 'POST'])
def get_churn_prediction():
    if request.method == 'POST':
        form = request.form
        id = form.get('id')
        customer = form.get('customer')
        surname = form.get('surname')
        score = form.get('score')
        geography = form.get('geography')
        gender = form.get('gender')
        age = form.get('age')
        tenure = form.get('tenure')
        balance = form.get('balance')
        products = form.get('products')
        crcard = form.get('crcard')
        activemember = form.get('activemember')
        salary = form.get('salary')
    data = {'RowNumber': [id],
            'CustomerId': [customer],
            'Surname': [surname],
            'CreditScore': [score],
            'Geography': [geography],
            'Gender': [gender],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [products],
            'HasCrCard': [crcard],
            'IsActiveMember': [activemember],
            'EstimatedSalary': [salary]}
    model = load_model(model_path='models/gboosting.pkl')
    prediction = predict(model=model, data=data)
    churn = 'Let him go' if prediction.get("prediction") == 1 else 'remain a user'
    # return f'<h1>The user {surname} is likeky to {churn}'
    return render_template("prediction.html",
                           surname=surname,
                           churn=churn)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3001, debug=True)
