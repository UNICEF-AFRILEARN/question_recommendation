from flask import Flask, render_template, request,jsonify
from flask_cors import CORS
from bson import ObjectId
from recommender import get_recommendations

app = Flask(__name__)
CORS(app, resources={r"*":{"origins":"*"}})

@app.route('/')
def main():
    return render_template('questions.html')

@app.route('/recommend', methods=['GET','POST'])
def reco_system():
    """
    The `reco_system` function is a recommendation system that takes in user data and returns a list of
    recommended questions.
    :return: a list of questions.
    """
    if request.method == 'GET':
        return jsonify({"response":"Send a POST request with userId, class_name, n_questions,rec_type"})
    else:
        data = request.get_json()
        if "userId" in data:
            userId = data['userId']
        else:
            userId = None
        class_name= data['class_name']
        n_questions = int(data['n_questions'])
        rec_type = data['rec_type']
        if "lessonId" in data:
            lessonId = data['lessonId']
        else:
            lessonId = None
        subject_name = data['subject_name']
        if userId:
            questions = get_recommendations(class_name,n_questions,[ObjectId(userId)],rec_type,lessonId,subject_name)
        else:
            questions = get_recommendations(class_name,n_questions,userId,rec_type,lessonId,subject_name)
        questions = [str(value) for value in questions]
        return questions

@app.route('/submit', methods=['POST', 'GET'])
def submit():
    """
    The `submit` function handles a form submission, retrieves the form data, and calls the
    `get_recommendations` function to get a list of recommended questions based on the form inputs.
    :return: either a list of questions or a message depending on the conditions.
    """
    if request.method == 'POST' or request.method == 'GET':
        class_name = request.form['class_name']
        n_questions = int(request.form['n_questions'])
        userId = request.form['userId']
        rec_type = request.form['rec_type']
        lessonId = request.form['lessonId']
        subject_name = request.form['subject_name']
        if (class_name == '') or (n_questions == ''):
            return render_template('questions.html', message='Please enter required fields')
        else:
            if userId:
                questions = get_recommendations(class_name,n_questions,[ObjectId(userId)],rec_type,lessonId,subject_name)
            else:
                questions = get_recommendations(class_name,n_questions,userId,rec_type,lessonId,subject_name)
            return questions
    else:
        return render_template('index.html', message='Please enter required fields')
    
if __name__ == '__main__':
    app.debug = False
    app.run()