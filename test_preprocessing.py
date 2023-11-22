import pandas as pd
import os
from unittest.mock import patch, MagicMock
from .data_processing import get_classes, get_training_data, get_synthetic_data, get_user_data
from .data_processing import label_transforms, classification

from configparser import ConfigParser

config_object = ConfigParser()
config_object.read("config.ini")

reco_cluster = config_object["RECOSYSTEM"]
main_cluster = config_object["MONGODB"]

sample_lesson_map = pd.DataFrame({
    'lessonId': ['6012d297cfe09249249f89d4'],
    'courseId': ['5fff72b3de0bdb47f826feaf'],
    'title': ['title1'],
    'subjectId': ['60119a4f31c66a2ebd9eb4d2']
})

sample_user_activities = pd.DataFrame({
    'lessonId': ['6012d297cfe09249249f89d4'],
    'userId': ['601604c044b43c48f8a3a20a']
})

sample_questions = pd.DataFrame({
    '_id': ['647d842c174362c1980329db'],
    'courseId': ['5fff72b3de0bdb47f826feaf']
})

# Use decorators to mock os.environ and MongoClient
@patch.dict(os.environ, {'MAINDB_KEY': 'mock_key'})
@patch('pymongo.MongoClient')
def test_get_classes(mongo_client_mock):
    mongo_client_mock.return_value.afrilearn = MagicMock()
    with patch('pandas.read_csv', return_value=sample_lesson_map):
        result = get_classes(sample_user_activities)
    assert not result.empty
    assert 'courseId' in result.columns

@patch('pymongo.MongoClient')
def test_get_user_data(mongo_client_mock):
    #mongo_client_mock.return_value.afrilearn = MagicMock()
    #mongo_client_mock.return_value.afrilearn.aiquestionslight.find.return_value = sample_questions.to_dict(orient='records')
    #mongo_client_mock.return_value.afrilearn.recentactivities.find.return_value = sample_user_activities.to_dict(orient='records')
    
    userIds, questions_df, questions = get_user_data('5fff72b3de0bdb47f826feaf')
    
    assert userIds == ['601604c044b43c48f8a3a20a']
    assert not questions_df.empty
    assert not questions.empty

# def test_get_synthetic_data(mocker):
#     mocker.patch('your_module.get_user_data', return_value=(['user1'], sample_questions, sample_questions))
#     result = get_synthetic_data('course1')
#     assert not result.empty
#     assert 'userId' in result.columns

def test_label_transforms():
    df = pd.DataFrame({
        'difficulty': ['easy'],
        'subjectId': ['60119a4f31c66a2ebd9eb4d2'],
        'userId': ['601604c044b43c48f8a3a20a'],
        'questionId': ['647d842c174362c1980329db']
    })
    transformed_df, _, _, _, _ = label_transforms(df)
    assert 'difficulty' in transformed_df.columns
    assert transformed_df['difficulty'].iloc[0] == 0

def test_get_training_data():
    df = pd.DataFrame({
        'difficulty': ['easy'],
        'subjectId': ['60119a4f31c66a2ebd9eb4d2'],
        'userId': ['601604c044b43c48f8a3a20a'],
        'questionId': ['647d842c174362c1980329db'],
        'next_attempt': [1]
    })
    x_train, x_test, y_train, y_test = get_training_data(df)
    assert not x_train.empty
    assert not y_train.empty

def test_classification():
    df = pd.DataFrame({
        'difficulty': ['easy'],
        'subjectId': ['60119a4f31c66a2ebd9eb4d2'],
        'userId': ['601604c044b43c48f8a3a20a'],
        'questionId': ['647d842c174362c1980329db'],
        'next_attempt': [1]
    })
    classifier, accuracy = classification(df)
    assert hasattr(classifier, 'predict')
    assert isinstance(accuracy, float)

