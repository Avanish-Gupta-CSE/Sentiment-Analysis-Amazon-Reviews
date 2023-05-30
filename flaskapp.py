# from flask import Flask, render_template, request
# from wtforms import Form, TextAreaField, validators
# import pickle
# import sqlite3
# import os
# import numpy as np
# import joblib
# loaded_model = joblib.load('./pkl_objects/model.pkl')

# loaded_stop = joblib.load('./pkl_objects/stopwords.pkl')
# loaded_vec = joblib.load('./pkl_objects/vectorizer.pkl')
# app = Flask(__name__)


# def classify(document):
#     label = {0: 'negative', 1: 'positive'}
#     X = loaded_vec.transform([document])
#     y = loaded_model.predict(X)[0]
#     proba = np.max(loaded_model.predict_proba(X))
#     return (label[y], proba)


# class ReviewForm(Form):

#     review = TextAreaField('', [validators.DataRequired(),validators.length(min=15)])


# @app.route('/')
# def index():
#     form = ReviewForm(request.form)
#     return render_template('reviewform.html', form=form)


# @app.route('/results', methods=['POST'])
# def results():
#     form = ReviewForm(request.form)
#     if request.method == 'POST' and form.validate():
#         review = request.form['review']
#         (y, proba) = classify(review)
#         return render_template('results.html', content=review,prediction=y, probability=round(proba* 100, 2))
#     return render_template('reviewform.html', form=form)
# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import sqlite3
import os
import numpy as np
import joblib
# loaded_model1 = joblib.load('./pkl_objects/model-logistic-regresion.pkl')
# loaded_model2= joblib.load('./pkl_objects/model-random-forest.pkl')
# loaded_model3 = joblib.load('./pkl_objects/model-xg-boost.pkl')
# models = [loaded_model1,loaded_model2,loaded_model3]
loaded_model = joblib.load('./pkl_objects/model-logistic-regresion.pkl')
models = [loaded_model]

loaded_stop = joblib.load('./pkl_objects/stopwords.pkl')
loaded_vec = joblib.load('./pkl_objects/vectorizer.pkl')
app = Flask(__name__)


def classify(document, models):
    labels = {0: 'negative', 1: 'positive'}
    X = loaded_vec.transform([document])
    results = []
    for model in models:
        y = model.predict(X)[0]
        proba = np.max(model.predict_proba(X))
        results.append((labels[y], proba))
    return results
# def classify(document):
#     label = {0: 'negative', 1: 'positive'}
#     X = loaded_vec.transform([document])
#     y = loaded_model.predict(X)[0]
#     proba = np.max(loaded_model.predict_proba(X))
#     return label[y], proba



class ReviewForm(Form):

    review = TextAreaField('', [validators.DataRequired(),validators.length(min=15)])


@app.route('/')
def index():
    form = ReviewForm(request.form)
    return render_template('reviewform.html', form=form)


@app.route('/results', methods=['POST'])
def results():
    form = ReviewForm(request.form)
    if request.method == 'POST' and form.validate():
        review = request.form['review']
        # models = [loaded_model1, loaded_model2, loaded_model3]
        models = [loaded_model]
        results = classify(review, models)
        return render_template('results.html', content=review, results=results)
    return render_template('reviewform.html', form=form)
# def results():
#     form = ReviewForm(request.form)
#     if request.method == 'POST' and form.validate():
#         review = request.form['review']
#         y, proba = classify(review)
#         return render_template('results.html',content=review,prediction=y,probability=round(proba*100, 2))
#     return render_template('reviewform.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)