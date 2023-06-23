from bson import ObjectId
import pickle



def get_prob_correct(model,unattempted_questions):
    X = unattempted_questions.drop(['next_attempt'],axis=1)
    prob_correct = model.predict_proba(X)[:,1]
    return prob_correct

def get_recommendations(userId,class_name,n_questions,rec_type=None,lessonId=None,subject_name=None):
    class_name_dict={"JSS One":0,"JSS Two": 1, "JSSS Three": 2,
                    "SSS One": 3, "SSS Two":4, "SSS Three":5}
    class_label = class_name_dict[class_name]
    
    responses = pickle.load(open('responses.pkl','rb'))
    responses_df = responses["responses_df"+str(class_label)]
    
    label_transforms =  pickle.load(open('encoders.pkl','rb'))
    label_encoders = label_transforms["encoder"+str(class_label)]
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
            subject_recommendation_df= unattempted_questions[unattempted_questions["subject_name"]==subject_name]
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
 