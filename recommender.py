from bson import ObjectId
import pandas as pd
import pickle



def get_prob_correct(model,unattempted_questions):
    X = unattempted_questions.drop(['next_attempt'],axis=1)
    prob_correct = model.predict_proba(X)[:,1]
    return prob_correct

def get_recommendations(userId,courseId,n_questions,rec_type=None,lessonId=None,subject_name=None):
    classid_dict={"5fff72b3de0bdb47f826feaf":0,"5fff7329de0bdb47f826feb0": 1, "5fff734ade0bdb47f826feb1": 2,
                    "5fff7371de0bdb47f826feb2": 3, "5fff7380de0bdb47f826feb3":4, "5fff7399de0bdb47f826feb4":5}
    class_label = classid_dict[courseId]
    
    responses_df = pd.read_parquet('responses.parquet')
    responses_df = responses_df[responses_df["course_Id"]==courseId]

    responses_df = responses_df.drop(["course_Id"],axis=1)
    
    label_transforms =  pickle.load(open('encoders.pkl','rb'))
    label_encoders = label_transforms["label"+str(class_label)]
    le_userId = label_encoders[2]
    userId_encoded = le_userId.transform([userId])[0]
    
    unattempted_questions = responses_df[(responses_df["userId"]==userId_encoded)&(responses_df["number_of_attempts"]==0)]
    if len(unattempted_questions)==0:
        unattempted_questions = responses_df[(responses_df["userId"]==userId_encoded)]

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
            print(subject_recommendation_df)
            recommended_questions = subject_recommendation_df['questionId'].values[:n_questions]
        else:
            return print("Enter a subject name to get question recommendations for this subject.")
    else:
        unattempted_questions.sort_values(by="prob_correct",ascending=False,inplace = True)
        recommended_questions = unattempted_questions['questionId'].values[:n_questions]
    recommended_questions  = le_questionId.inverse_transform(recommended_questions)
    return recommended_questions
 