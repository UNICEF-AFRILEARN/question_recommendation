import pandas as pd
import numpy as np
from pymongo import MongoClient
from configparser import ConfigParser
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

config_object = ConfigParser()
config_object.read("config.ini")
connection_details = config_object["MONGODB"]
cluster = eval(connection_details["cluster"])

client = MongoClient(cluster)
db=client.afrilearn

def get_classes(df):
    lesson_map = pd.read_csv("lessonid_class_map.csv",index_col=[0])
    merged = lesson_map[['lessonId','class_name','title','subject_name']].merge(df,on='lessonId',how='right')
    return merged

def get_user_data(class_name):
    """
    This function retrieves user data and questions related to a specific class name.
    
    :param class_name: The name of the class for which we want to retrieve user data and questions
    :return: The function `get_user_data` returns two objects: `userIds` and `questions_df`. `userIds`
    is a list of user IDs who belong to the specified class name, and `questions_df` is a pandas
    DataFrame containing the question IDs of all questions that belong to the specified class name.
    """
    questions = pd.DataFrame(list(db.aiquestions.find()))
    user_activities =  pd.DataFrame(list(db.recentactivities.find()))
    user_activities['lessonId'] = user_activities['lessonId'].astype(str)
    user_classes = get_classes(user_activities)
    grouped_df = user_classes[['userId','class_name']].groupby("userId")['class_name'].agg(list).reset_index()
    grouped_df['user_class_name'] =grouped_df['class_name'].apply(lambda x: list(set(x))[0])
    userid_classes = grouped_df[['userId','user_class_name']] 
    questions_df = pd.DataFrame({'questionId':questions_with_class[questions_with_class['class_name']==class_name]['_id'].values})
    userIds = list(userid_classes[userid_classes['user_class_name']==class_name]['userId'].values)

    questions_with_class = get_classes(questions)
    questions_with_class.to_pickle("questions.pkl")
    return userIds, questions_df

def get_synthetic_data(class_name):
    userIds,questions_df,questions_with_class = get_user_data(class_name)
    data = []
    # Generate synthetic data
    for user in userIds:
        for _, question in questions_df.iterrows():
            attempted = np.random.randint(0, 2)
            if attempted==1:
                correct=[]
                for count in range(6):
                    correct.append(np.random.randint(0, 2))
                correct.append(np.random.randint(sum(correct),sum(correct)+3))
                # Append the generated data to the list
                data.append([user, question['questionId']]+correct)
            else:
                correct=[]
                for count in range(7):
                    correct.append(0)
                # Append the generated data to the list
                data.append([user, question['questionId']]+correct)

    # Convert the list to a dataframe
    responses_df = pd.DataFrame(data, columns=['userId', 'questionId', 'first_attempt', 'second_attempt',
                                               'third_attempt', 'fourth_attempt', 'fifth_attempt','next_attempt','number_of_attempts'])
    
    col_list = ['first_attempt', 'second_attempt','third_attempt', 'fourth_attempt', 'fifth_attempt']
    responses_df['pass_ratio'] = responses_df[col_list].sum(axis=1)/responses_df['number_of_attempts']
    responses_df['pass_ratio'].fillna(0,inplace=True)
    questions_columns = ['class_name', 'title', 'subject_name', '_id', 'question','options', 'difficulty']
    responses_df=responses_df.merge(questions_with_class[questions_columns], left_on='questionId',right_on='_id',how='left')
    responses_df=responses_df.drop(['class_name','_id','title','question','options'],axis=1)
    return responses_df

def label_transforms(responses_df):
    le_difficulty = LabelEncoder()
    responses_df['difficulty'] = le_difficulty.fit_transform(responses_df['difficulty'])
    le_subject_name= LabelEncoder()
    responses_df['subject_name'] = le_subject_name.fit_transform(responses_df['subject_name'])
    le_userId= LabelEncoder()
    responses_df['userId'] = le_userId.fit_transform(responses_df['userId'])
    le_questionId = LabelEncoder()
    responses_df['questionId'] = le_questionId.fit_transform(responses_df['questionId'])
    return responses_df, le_difficulty, le_subject_name, le_userId,le_questionId

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

 
    