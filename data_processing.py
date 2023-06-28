import pandas as pd
from pymongo import MongoClient
from configparser import ConfigParser
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import random

config_object = ConfigParser()
config_object.read("config.ini")
connection_details = config_object["MONGODB"]
cluster = eval(connection_details["cluster"])

client = MongoClient(cluster)
db=client.afrilearn

def get_classes(df):
    lesson_map = pd.read_csv("lessonid_class_map.csv",index_col=[0])
    merged = lesson_map[['lessonId','courseId','title','subjectId']].merge(df,on='lessonId',how='right')
    return merged

def get_user_data(courseId):
    """
    This function retrieves user data and questions related to a specific class name.
    
    :param class_name: The name of the class for which we want to retrieve user data and questions
    :return: The function `get_user_data` returns two objects: `userIds` and `questions_df`. `userIds`
    is a list of user IDs who belong to the specified class name, and `questions_df` is a pandas
    DataFrame containing the question IDs of all questions that belong to the specified class name.
    """
    questions = pd.DataFrame(list(db.aiquestionslight.find()))
    user_activities =  pd.DataFrame(list(db.recentactivities.find()))
    user_activities['lessonId'] = user_activities['lessonId'].astype(str)
    user_classes = get_classes(user_activities)
    grouped_df = user_classes[['userId','courseId']].groupby("userId")['courseId'].agg(list).reset_index()
    grouped_df['user_courseId'] =grouped_df['courseId'].apply(lambda x: list(set(x))[0])
    userid_classes = grouped_df[['userId','user_courseId']] 
    questions_df = pd.DataFrame({'questionId':questions[questions['courseId']==courseId]['_id'].values})
    userIds = list(userid_classes[userid_classes['user_courseId']==courseId]['userId'].values)

    questions.to_pickle("questions.pkl")
    return userIds, questions_df,questions

def get_synthetic_data(class_name):
    userIds,questions_df,questions_with_class = get_user_data(class_name)
    questionIds = list(questions_df['questionId'].values)
    # Create an empty list to store the generated data
    synthetic_data = list()

    # Generate synthetic data
    for user_id in userIds:
        for question_id in questionIds:
            attempted = random.choice([0, 1])
            success = list()
            if attempted == 1:
                attempts = random.randint(3, 5)  # Number of attempts for the question
                next_attempt = random.choice([0, 1])
                
                for i in range(1, 6):
                    if i <= attempts:
                        success.append(random.choice([0, 1]))
                    else:
                        success.append(0)
            else:
                attempts = 0
                success = [0]*5
                next_attempt = random.choice([0, 1])

            synthetic_data.append({"userId":user_id,"questionId":question_id,
                                   "first_attempt":success[0],"second_attempt":success[1],
                                   "third_attempt":success[2],
                                   "fourth_attempt": success[3],"fifth_attempt":success[4],
                                   "number_of_attempts":attempts,"next_attempt":next_attempt})
    responses_df = pd.DataFrame(synthetic_data)
    col_list = ['first_attempt', 'second_attempt','third_attempt', 'fourth_attempt', 'fifth_attempt']
    responses_df['pass_ratio'] = responses_df[col_list].sum(axis=1)/responses_df['number_of_attempts']
    responses_df.fillna(0,inplace=True)
    questions_columns = ['courseId', 'title', 'subjectId', '_id', 'question','options', 'difficulty']
    responses_df=responses_df.merge(questions_with_class[questions_columns], left_on='questionId',right_on='_id',how='left')
    responses_df=responses_df.drop(['courseId','_id','title','question','options'],axis=1)
    responses_df['next_attempt'] = responses_df.apply(lambda x: 1 if (x['pass_ratio']>0.5) and (x['next_attempt']==0) else x['next_attempt'],axis=1)
    return responses_df

def label_transforms(responses_df):
    le_difficulty = LabelEncoder()
    responses_df['difficulty'] = le_difficulty.fit_transform(responses_df['difficulty'])
    le_subjectId= LabelEncoder()
    responses_df['subjectId'] = le_subjectId.fit_transform(responses_df['subjectId'])
    le_userId= LabelEncoder()
    responses_df['userId'] = le_userId.fit_transform(responses_df['userId'])
    le_questionId = LabelEncoder()
    responses_df['questionId'] = le_questionId.fit_transform(responses_df['questionId'])
    return responses_df, le_difficulty, le_subjectId, le_userId,le_questionId

def get_training_data(responses_df):
    responses_df,_,_,_,_ =label_transforms(responses_df)
    X = responses_df.drop(['next_attempt'],axis=1)
    Y = responses_df[['next_attempt']]
    
    x_train,x_test,y_train,y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test


def classification(responses_df):
    x_train,x_test,y_train,y_test = get_training_data(responses_df)
    classifier = LogisticRegression()
    classifier.fit(x_train,y_train)
    # Make predictions on the test set
    y_pred = classifier.predict(x_test)

    # Evaluate the accuracy of the classifier
    accuracy = accuracy_score(y_test, y_pred)
    return classifier,accuracy

 
    