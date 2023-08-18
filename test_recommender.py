import pytest
import pandas as pd
from bson import ObjectId
from sklearn.linear_model import LogisticRegression
from recommender import get_prob_correct, get_recommendations


@pytest.fixture(scope="module")
def sample_data():
    unattempted_questions = pd.DataFrame({
        "userId": [1, 2, 3],
        "questionId": [4, 5, 6],
        "next_attempt": [0, 0, 0],
    })
    return unattempted_questions


def test_get_prob_correct(sample_data):
    model = LogisticRegression()
    unattempted_questions = sample_data.copy()
    prob_correct = get_prob_correct(model, unattempted_questions)
    assert len(prob_correct) == len(unattempted_questions)


def test_get_recommendations():
    courseId = "5fff72b3de0bdb47f826feaf"
    n_questions = 5
    userId = "user1"
    rec_type = "lesson"
    lessonId = str(ObjectId())
    recommended_questions = get_recommendations(
        courseId, n_questions, userId=userId, rec_type=rec_type, lessonId=lessonId
    )
    assert len(recommended_questions) == n_questions

    rec_type = "subject"
    subject_name = "Math"
    recommended_questions = get_recommendations(
        courseId, n_questions, userId=userId, rec_type=rec_type, subject_name=subject_name
    )
    assert len(recommended_questions) == n_questions

    rec_type = None
    recommended_questions = get_recommendations(courseId, n_questions)
    assert len(recommended_questions) == n_questions

