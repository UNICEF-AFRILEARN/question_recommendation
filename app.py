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
    if request.method == 'GET':
        return jsonify({"response":"Send a POST request with userId, class_name, n_questions,rec_type"})
    else:
        data = request.get_json()
        userId = data['userId']
        class_name= data['class_name']
        n_questions = int(data['n_questions'])
        rec_type = data['rec_type']
        lessonId = data['lessonId']
        subject_name = data['subject_name']
        questions =get_recommendations([ObjectId(userId)],class_name,n_questions,rec_type,lessonId,subject_name)
        return questions

@app.route('/submit', methods=['POST', 'GET'])
def submit():
    if request.method == 'POST' or request.method == 'GET':
        userId = request.form['userId']
        class_name = request.form['class_name']
        n_questions = int(request.form['n_questions'])
        rec_type = request.form['rec_type']
        lessonId = request.form['lessonId']
        subject_name = request.form['subject_name']
        if (class_name == '') or (n_questions == ''):
            return render_template('index.html', message='Please enter required fields')
        else:
            if userId:
                questions = get_recommendations(class_name,n_questions,[ObjectId(userId)],rec_type,lessonId,subject_name)
            else:
                questions = get_recommendations(class_name,n_questions,userId,rec_type,lessonId,subject_name)
            questions = [str(value) for value in questions]
            return questions
    else:
        return render_template('index.html', message='Please enter required fields')
    
if __name__ == '__main__':
    app.debug = False
    app.run()