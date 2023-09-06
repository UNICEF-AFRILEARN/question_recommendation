from bson import ObjectId
import pandas as pd
import pickle
import random
import os
from pymongo import MongoClient

reco_cluster = os.environ['RECODB_KEY']
main_cluster = os.environ['MAINDB_KEY']

client = MongoClient(reco_cluster)
db = client.afrilearn

main_client = MongoClient(main_cluster)
maindb = main_client.afrilearn

def get_prob_correct(model,unattempted_questions:pd.DataFrame)->list:
    """
    The function `get_prob_correct` takes a machine learning model and a dataframe of unattempted
    questions as input, and returns a list of the predicted probabilities of getting the questions
    correct.
    
    :param model: The "model" parameter refers to a trained machine learning model that can predict the
    probability of a correct answer given certain features of a question. It could be any model that has
    a "predict_proba" method, such as a logistic regression model or a random forest classifier
    :param unattempted_questions: The parameter "unattempted_questions" is a pandas DataFrame that
    contains the data for the unattempted questions. It should have the following columns:
    :type unattempted_questions: pd.DataFrame
    :return: a list of probabilities of getting the correct answer for each unattempted question.
    """
    X = unattempted_questions.drop(['next_attempt'],axis=1)
    prob_correct = model.predict_proba(X)[:,1]
    return prob_correct

def get_recommendations(courseId:str,n_questions:int,userId:str=None,rec_type:str=None,lessonId:str=None,subject_name:str=None)->list:
    """
    The function `get_recommendations` takes in various parameters such as courseId, n_questions,
    userId, rec_type, lessonId, and subject_name, and returns a list of recommended questions based on
    the given inputs.
    
    :param courseId: The `courseId` parameter is a string that represents the ID of the course for which
    you want to get recommendations
    :type courseId: str
    :param n_questions: The parameter `n_questions` specifies the number of question recommendations
    that you want to retrieve
    :type n_questions: int
    :param userId: The `userId` parameter is used to specify the user for whom the recommendations are
    being generated. It is an optional parameter, so if it is not provided, the function will generate
    random recommendations instead of personalized recommendations
    :type userId: str
    :param rec_type: The `rec_type` parameter is used to specify the type of recommendation you want. It
    can have two possible values:
    :type rec_type: str
    :param lessonId: The `lessonId` parameter is used to specify the ID of a lesson for which you want
    to get question recommendations
    :type lessonId: str
    :param subject_name: The `subject_name` parameter is used to specify the name of the subject for
    which you want to get question recommendations
    :type subject_name: str
    :return: The function `get_recommendations` returns a list of recommended question IDs.
    """
    classid_dict={"5fff72b3de0bdb47f826feaf":0,"5fff7329de0bdb47f826feb0": 1, "5fff734ade0bdb47f826feb1": 2,
                    "5fff7371de0bdb47f826feb2": 3, "5fff7380de0bdb47f826feb3":4, "5fff7399de0bdb47f826feb4":5}
    class_label = classid_dict[courseId]

    if userId:         
        label_transforms =  pickle.load(open('encoders.pkl','rb'))
        label_encoders = label_transforms["label"+str(class_label)]
        
        le_userId = label_encoders[2]
        userId_encoded = le_userId.transform([userId])[0]
        unattempted_questions =pd.DataFrame(list(db.studentresponses.find({"course_Id":courseId,"userId":int(userId_encoded),"number_of_attempts":0})))

        if len(unattempted_questions)==0:
            unattempted_questions = pd.DataFrame(list(db.studentresponses.find({"course_Id":courseId,"userId":int(userId_encoded)})))
        if len(unattempted_questions)==0:
            raise ValueError("This userId and classId combination is not in the database.")
        unattempted_questions= unattempted_questions.drop(["_id","course_Id"],axis=1)

        models =  pickle.load(open('classifiers.pkl','rb'))
        model = models["classifier"+str(class_label)]
        unattempted_questions["prob_correct"] = get_prob_correct(model, unattempted_questions)
        
        le_subjectname = label_encoders[1]
        le_questionId = label_encoders[3]
        if rec_type=="lesson":
            if lessonId:
                unattempted_questions["questionId"] = unattempted_questions["questionId"].apply(lambda x: le_questionId.inverse_transform([x]))
                questions = pickle.load(open('questions.pkl','rb'))
                lesson_recommendation_df = questions.merge(unattempted_questions,left_on=['_id'],right_on=['questionId'],how='right')
                lesson_recommendation_df= lesson_recommendation_df[lesson_recommendation_df["lessonId"]==ObjectId(lessonId)]
                lesson_recommendation_df.sort_values(by="prob_correct",ascending=False,inplace = True)
                recommended_questions = lesson_recommendation_df['questionId'].values[:n_questions]
            else:
                return print("Enter a lessonId to get question recommendations for this lesson.")
        elif rec_type=="subject":
            if subject_name:
                subject_name = le_subjectname.transform([subject_name])[0]
                subject_recommendation_df= unattempted_questions[unattempted_questions["subjectId"]==subject_name]
                subject_recommendation_df.sort_values(by="prob_correct",ascending=False,inplace = True)
                recommended_questions = subject_recommendation_df['questionId'].values[:n_questions]
            else:
                return print("Enter a subject name to get question recommendations for this subject.")
        else:
            unattempted_questions.sort_values(by="prob_correct",ascending=False,inplace = True)
            recommended_questions = unattempted_questions['questionId'].values[:n_questions]
        recommended_questions  = list(le_questionId.inverse_transform(recommended_questions))
    else:
        questions = pickle.load(open('questions.pkl','rb'))
        questions = questions[questions['subjectId']==subject_name]
        recommended_questions=random.choices(list(questions['_id'].unique()),k=n_questions)
    recommended_questions = pd.DataFrame(list(maindb.aiquestionslight.find({'_id':{"$in":recommended_questions}})))
    return recommended_questions.to_dict("records") 