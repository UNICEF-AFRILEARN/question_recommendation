#!/usr/bin/env python
# coding: utf-8

# # define sythetic data requirement
# * userid
# * questionid
# * tfid of questions and options
# * difficulty levels
# * last five attempts
# * pass ratio
# * number of attempts
# * class
# * subject

# In[1]:


# import modules


# In[2]:


from importlib import reload
import data_processing
reload(data_processing)
# load synthetic data from another notebook
from data_processing import *
import pandas as pd
import pickle


# In[3]:


# set up loop to get responses and models


# In[4]:


courseids = ['5fff72b3de0bdb47f826feaf','5fff7329de0bdb47f826feb0','5fff734ade0bdb47f826feb1',
             '5fff7371de0bdb47f826feb2','5fff7380de0bdb47f826feb3','5fff7399de0bdb47f826feb4']


# In[5]:


questions = pickle.load(open('questions.pkl','rb'))


# In[6]:


questions


# In[8]:


df = pd.read_parquet("converter.parquet")


# In[11]:


df[(df['class_name']=='sss-2')&(df['subject_name']=='physics')]


# In[5]:


responses_dict = {}
classifier_dict = {}
accuracy_dict = {}
encoders_dict = {}
for index,classid in enumerate(courseids):
    responses_df = get_synthetic_data(classid)
    responses_df, le_difficulty, le_subject_name, le_userid,le_questionId = label_transforms(responses_df)
    responses_dict['responses_df' + str(index)] = responses_df
    encoders_dict["label"+str(index)] = [le_difficulty, le_subject_name, le_userid,le_questionId]
    classifier,accuracy = classification(responses_df)
    classifier_dict["classifier"+str(index)]=classifier
    accuracy_dict["accuracy"+str(index)] = accuracy


# In[6]:


accuracy_dict


# In[9]:


all_responses = pd.DataFrame()
for index in range(6):
    responses = responses_dict['responses_df' + str(index)]
    responses['course_Id'] = courseids[index]
    if len(all_responses)==0:
        all_responses = responses
    else:
        all_responses = pd.concat([all_responses,responses])


# In[11]:


all_responses.to_parquet("responses.parquet")


# In[5]:


df = pd.read_parquet("responses.parquet")


# In[6]:


df


# In[16]:


filename = "classifiers.pkl"
s = pickle.dump(classifier_dict, open(filename, 'wb'))


# In[17]:


filename = "accuracy.pkl"
s = pickle.dump(accuracy_dict, open(filename, 'wb'))


# In[18]:


filename = "encoders.pkl"
s = pickle.dump(encoders_dict, open(filename, 'wb'))


# In[8]:


# save to database


# In[7]:


config_object = ConfigParser()
config_object.read("config.ini")
connection_details = config_object["RECOSYSTEM"]
cluster = eval(connection_details["cluster"])


# In[8]:


cluster


# In[9]:


recocluster = "mongodb+srv://afrilearn_ai:l3pLnbLCzpHwkwIH@recommendersystem.enhrx0m.mongodb.net/?retryWrites=true&w=majority"
recoclient = MongoClient(recocluster)
recodb = recoclient.afrilearn
recocollection=recodb.studentresponses


# In[10]:


recocollection


# In[ ]:


JSS1 649d51f9afcc3a9fedc61f87


# In[29]:


from configparser import ConfigParser
config_object = ConfigParser()
config_object.read("config.ini")

reco_cluster = config_object["RECOSYSTEM"]
main_cluster = eval(config_object.get("MONGODB",'cluster'))


# In[32]:


client = MongoClient(main_cluster)


# In[33]:


maindb = client.afrilearn


# In[39]:


query_result = maindb.aiquestionslight.find(
    {'subject_id': '5fff7329de0bdb47f826feb0', 'course_id':'60119ae731c66a2ebd9eb4dc'}  # Projection to include only subject_id and course_id
)


# In[41]:


pd.DataFrame(list(query_result))


# In[38]:


converter[converter['class_name']=='jss-2']


# In[47]:


query_result = maindb.aiquestionslight.find()


# In[48]:


quest = pd.DataFrame(list(query_result))


# In[58]:


quest


# In[63]:


quest[(quest['courseId']=='5fff7329de0bdb47f826feb0')&(quest['subjectId']=='60119ae731c66a2ebd9eb4dc')]


# In[60]:


converter.loc[converter['subject_name'] == str('agricultural-science'), 'old_subjectId'].values[0]


# In[62]:


converter.loc[(converter['subject_name'] == str('agricultural-science'))&(converter['class_name'] == str('jss-2')), 'old_subjectId'].values[0]


# In[56]:


ques=pd.concat([questions,quest])


# In[57]:


ques.drop_duplicates()


# In[141]:


eval(main_cluster)


# In[38]:


cluster = "mongodb+srv://afrilearn_ai:l3pLnbLCzpHwkwIH@afrilearn.tjk9n.mongodb.net/?retryWrites=true&w=majority"


# In[46]:


client = MongoClient(cluster)
recodb = client["afrilearn-prod"]


# In[41]:


classes = recodb.classlevels


# In[43]:


cursor = classes.find()


# In[85]:


classes= pd.DataFrame(list(recodb.classlevels.find()))


# In[86]:


classes


# In[83]:


subjects= pd.DataFrame(list(recodb.subjects.find()))


# In[52]:


subjects = subjects[['_id','name']]


# In[84]:


subjects


# In[88]:


classes = classes.rename(columns={'_id':'classId'})


# In[91]:


classes = classes.rename(columns={'name':'class_name'})


# In[92]:


subjects = subjects.rename(columns={'name':'subject_name'})


# In[94]:


subjects = subjects.rename(columns={'_id':'subjectId'})


# In[95]:


combined_table = classes.merge(subjects, left_on='groupId',right_on='groupId',how='inner')


# In[102]:


combined_table = combined_table.rename(columns={'subjectId':'old_subjectId'})


# In[98]:


df_subject_class['class_name'] = df_subject_class['class_name'].str.replace('One','1')
df_subject_class['class_name'] = df_subject_class['class_name'].str.replace('Two','2')
df_subject_class['class_name'] = df_subject_class['class_name'].str.replace('Three','3')


# In[103]:


df_subject_class = df_subject_class.rename(columns={'subjectId':'new_subjectId'})


# In[105]:


final_table = combined_table.merge(df_subject_class, left_on=['class_name','subject_name'],right_on=['class_name','subject_name'],how='inner')


# In[108]:


final_table = final_table.rename(columns={'courseId':'old_courseId','classId':'new_courseId'})


# In[111]:


final_table = final_table[['new_courseId','class_name','old_subjectId','subject_name', 'new_subjectId','old_courseId']]


# In[113]:


final_table=final_table.rename(columns={'new_subjectId':'old_subjectId','old_subjectId':'new_subjectId'})


# In[114]:


final_table[final_table['old_courseId']=='5fff72b3de0bdb47f826feaf']


# In[116]:


final_table['new_courseId'] = final_table['new_courseId'].astype(str)


# In[117]:


final_table['old_courseId'] = final_table['old_courseId'].astype(str)
final_table['new_subjectId'] = final_table['new_subjectId'].astype(str)
final_table['old_subjectId'] = final_table['old_subjectId'].astype(str)


# In[118]:


final_table.to_parquet("converter.parquet")


# In[110]:


df_subject_class


# In[27]:


questions = pickle.load(open('questions.pkl','rb'))


# In[120]:


subjectId = '60119a3331c66a2ebd9eb4cf'


# In[121]:


courseId ='5fff72b3de0bdb47f826feaf'


# In[124]:


questions = questions[(questions['subjectId']==subjectId) & (questions['courseId']==courseId)]


# In[126]:


recommended_questions=random.choices(list(questions['_id'].unique()),k=10)


# In[127]:


recommended_questions


# In[143]:


main_client = MongoClient(main_cluster)
maindb = main_client.afrilearn


# In[128]:


maindb=client.afrilearn


# In[144]:


recommended_questionsv = pd.DataFrame(list(maindb.aiquestionslight.find({'_id':{"$in":recommended_questions}})))


# In[145]:


recommended_questionsv


# In[131]:


recommended_questionsv['_id'].isin(recommended_questions)


# In[ ]:


{'_id':{"$in":recommended_questions}}


# In[31]:


for index in range(0,len(df),100000):
    recocollection.insert_many(all_responses[index:index+100000].to_dict('records'))
    print("records inserted")


# In[40]:


questions = pickle.load(open('questions.pkl','rb'))
recommended_questions=random.choices(list(questions['_id'].unique()),k=10)


# In[42]:


recommended_questions = [str(question) for question in recommended_questions]


# In[43]:


recommended_questions


# In[ ]:


# test directly on heroku


# In[37]:


get_ipython().run_cell_magic('time', '', 'unattemptedquestions=pd.DataFrame(list(recodb.studentresponses.find({"course_Id":"5fff72b3de0bdb47f826feaf","userId":0,"number_of_attempts":0})))\n')


# In[38]:


unattemptedquestions


# In[33]:


df["response"]=df.apply(lambda row: {"first_attempt":row['first_attempt'],"second_attempt":row['second_attempt'],
                     "third_attempt":row['third_attempt'],"fourth_attempt":row['fourth_attempt'],"fifth_attempt":row['fifth_attempt'],"next_attempt":row['next_attempt'],"pass_ratio":row["pass_ratio"]},axis=1)


# In[36]:


df


# In[11]:


questions = pickle.load(open('questions.pkl','rb'))


# In[13]:


questions['courseId'].unique()


# In[24]:


subjectids = list(questions['subjectId'].unique())


# In[16]:


df = pd.read_csv('lessonid_class_map.csv')


# In[17]:


df[['courseId','class_name']]


# In[19]:


df_course_class = df[['courseId','class_name']]


# In[20]:


df_course_class['courseId'].unique()


# In[22]:


df_course_class = df_course_class[df_course_class['courseId'].isin(courseids)]


# In[23]:


df_course_class['courseId'].unique()


# In[69]:


df_subject_class = df[['subjectId','subject_name','courseId','class_name']]


# In[70]:


df_subject_class


# In[71]:


df_subject_class = df_subject_class[df_subject_class['subjectId'].isin(subjectids)]


# In[72]:


df_subject_class


# In[1]:


import pandas as pd


# In[18]:


converter = pd.read_parquet("converter.parquet")


# In[5]:


converter = converter.drop_duplicates()


# In[8]:


converter=converter.reset_index(drop=True)


# In[16]:


converter.to_parquet("converter.parquet")


# In[14]:


converter['class_name'] =converter['class_name'].str.lower().str.replace(' ','-')


# In[15]:


converter['subject_name'] =converter['subject_name'].str.lower().str.replace(' ','-')


# In[19]:


converter


# In[20]:


classid_dict={"5fff72b3de0bdb47f826feaf":0,"5fff7329de0bdb47f826feb0": 1, "5fff734ade0bdb47f826feb1": 2,
                "5fff7371de0bdb47f826feb2": 3, "5fff7380de0bdb47f826feb3":4, "5fff7399de0bdb47f826feb4":5}


# In[21]:


class_label = classid_dict['hjjh']


# In[77]:


df_subject_class['class_name'].str..replace('one','1')


# In[78]:


df_subject_class['class_name_updated'] = df_subject_class['class_name'].str.replace('One','1')
df_subject_class['class_name_updated'] = df_subject_class['class_name_updated'].str.replace('Two','2')
df_subject_class['class_name_updated'] = df_subject_class['class_name_updated'].str.replace('Three','3')
df_subject_class['class_name_updated'] =  df_subject_class['class_name_updated'].str.lower().str.replace(' ', '_')


# In[80]:


df_subject_class['class_name_updated'] =  df_subject_class['class_name_updated'].str.replace('_', '-')


# In[81]:


df_subject_class


# In[82]:


subjects


# In[ ]:





# In[28]:


questions


# In[ ]:





# In[ ]:





# In[37]:


df_subject_class = df_subject_class.drop_duplicates()


# In[56]:


df_subject_class[df_subject_class['subject_name']=='Agricultural Science']


# In[ ]:





# In[32]:


df_course_class['class_name_updated'] = df_course_class['class_name'].str.lower().str.replace(' ', '_')


# In[35]:


df_course_class = df_course_class.drop_duplicates()


# In[ ]:





# In[23]:


converter[converter['class_name']=='jss-2']


# In[24]:


questions = 


# In[ ]:





# In[ ]:


df.drop([])


# In[4]:


ranked_results_with_id['value'] = ranked_results_with_id.apply(lambda row: {'rank': row['rank'],
                                                                                'metric_name': row['metric_name'], 
                                                                                'metric':row[bestfit_object.error_metric]}, axis=1)
    ranked_results_with_id= ranked_results_with_id.drop(['MAE', 'rank','metric_name','model_type'], axis=1)


# In[5]:


df  = pd.read_csv("all_responses.csv")


# In[15]:


df[df['course_Id']=="5fff5a7ede0bdb47f826fea9"].to_csv("ssstwo.csv")


# In[16]:


df['course_Id'].value_counts()


# In[29]:


courseids = ['5fff72b3de0bdb47f826feaf','5fff7329de0bdb47f826feb0','5fff734ade0bdb47f826feb1','5fff7371de0bdb47f826feb2','5fff7380de0bdb47f826feb3','5fff7399de0bdb47f826feb4']


# In[48]:


lesson_map = pd.read_csv("lessonid_class_map.csv")


# In[50]:


lesson_map[lesson_map['subjectId']=="5fff5bab3fd2d54b08047c82"]


# In[30]:


df[df['course_Id'].isin(courseids)]


# In[51]:


all_responses['course_Id'].value_counts()


# In[ ]:





# In[ ]:


{'courseId': '5fff72b3de0bdb47f826feaf', 'class_name': 'JSS One'},
 {'courseId': '5fff7329de0bdb47f826feb0', 'class_name': 'JSS Two'},
 {'courseId': '5fff734ade0bdb47f826feb1', 'class_name': 'JSS Three'},
 {'courseId': '5fff7371de0bdb47f826feb2', 'class_name': 'SSS One'},
 {'courseId': '5fff7380de0bdb47f826feb3', 'class_name': 'SSS Two'},
 {'courseId': '5fff7399de0bdb47f826feb4', 'class_name': 'SSS Three'},


# In[28]:


lesson_map[['courseId','class_name']].drop_duplicates().to_dict("records")


# In[22]:


lesson_map[['courseId']].value_counts()


# In[23]:


lesson_map['class_name'].value_counts()


# In[12]:


import sqlite3
import pandas as pd


# In[13]:


conn = sqlite3.connect('responses.db') 


# In[14]:


df.to_sql('studentresponses', conn)


# In[ ]:


['5fff72b3de0bdb47f826feaf',
 '5fff7329de0bdb47f826feb0',
 '5fff734ade0bdb47f826feb1',
 '5fff7371de0bdb47f826feb2',
 '5fff7380de0bdb47f826feb3',
 '5fff7399de0bdb47f826feb4']


# In[32]:


aiquestions=pd.DataFrame(list(db.aiquestions.find()))


# In[36]:


questions_with_courseid = get_classes(aiquestions)


# In[38]:


questions = questions_with_courseid[questions_with_courseid['courseId'].isin(courseids)]


# In[39]:


questions['courseId'].value_counts()


# In[40]:


df = questions.groupby('courseId').sample(n=1000, random_state=42)


# In[41]:


df['courseId'].value_counts()


# In[42]:


collection = db.aiquestionslight


# In[43]:


collection.insert_many(df.to_dict('records'))


# In[19]:


respo_df = get_synthetic_data("5fff72b3de0bdb47f826feaf")


# In[106]:


respo_df['difficulty'].value_counts()


# In[111]:


respo_df['next_attempt'].value_counts()


# In[103]:


respo_df['next_attempt'] = respo_df.apply(lambda x: 1 if (x['pass_ratio']>0.5) and (x['next_attempt']==0) else x['next_attempt'],axis=1)


# In[ ]:


responses_df['next_attempt'] = responses_df.apply(lambda x: 1 if x['pass_ratio']>0.5 else 0)


# In[45]:


len(respo_df)


# In[28]:


random.choices(list(respo_df['questionId'].unique()),k=10)


# In[ ]:





# In[ ]:





# In[49]:


classes = ["JSS One","JSS Two","JSS Three","SSS One","SSS Two","SSS Three"]


# In[52]:


responses_df, df = get_synthetic_data("5fc8cfbb81a55b4c3c19737d")


# In[56]:


subjects = list(responses_df['subjectId'].values)


# In[20]:


lesson_map = pd.read_csv("lessonid_class_map.csv",index_col=[0])


# In[59]:


lesson_map[lesson_map['subjectId'].isin(subjects)]['subject_name'].value_counts()


# In[21]:


respo_df


# In[61]:


classes = list(lesson_map['courseId'].unique())


# In[63]:


classes = classes[:-1]


# In[71]:


responses_df


# In[ ]:


responses_df = get_synthetic_data(class_name)


# In[ ]:





# In[47]:


responses_dict = {}
classifier_dict = {}
accuracy_dict = {}
encoders_dict = {}
for index,class_name in enumerate(classes[6:12]):
    responses_df = get_synthetic_data(class_name)
    responses_df, le_difficulty, le_subject_name, le_userid,le_questionId = label_transforms(responses_df)
    responses_dict['responses_df' + str(index)] = responses_df


# In[48]:


responses_dict


# In[180]:


classesjss = classes[6:12]


# In[181]:


classesjss


# In[65]:


all_responses = pd.DataFrame()
for index in range(6):
    responses = responses_dict['responses_df' + str(index)]
    responses['course_Id'] = courseids[index]
    if len(all_responses)==0:
        all_responses = responses
    else:
        all_responses = pd.concat([all_responses,responses])


# In[66]:


len(all_responses)


# In[54]:


all_responses.to_csv("all_responses.csv")


# In[62]:


recocluster = "mongodb+srv://afrilearn_ai:l3pLnbLCzpHwkwIH@recommendersystem.enhrx0m.mongodb.net/?retryWrites=true&w=majority"
recoclient = MongoClient(recocluster)
recodb = recoclient.afrilearn
recocollection=recodb.studentresponses


# In[63]:


recodb = recoclient.afrilearn


# In[64]:


recocollection=recodb.studentresponses


# In[81]:


config_object = ConfigParser()
config_object.read("config.ini")
connection_details = config_object["MONGODB"]
cluster = eval(connection_details["cluster"])

client = MongoClient(cluster)
db=client.afrilearn


# In[55]:


collection = db.studentresponses


# In[60]:


all_responses.reset_index(drop=True, inplace=True)


# In[ ]:


for index in range(0,len(all_responses),100000):
    recocollection.insert_many(all_responses[index:index+100000].to_dict('records'))
    print("records inserted")


# In[85]:


all_responses


# In[86]:


accuracy_dict


# In[96]:


classes[9:10]


# In[98]:


df=all_responses[all_responses['course_Id'].isin(classes[9:10])]


# In[102]:


len(df)


# In[103]:


df[df["number_of_attempts"]!=0]


# In[104]:


all_responses


# In[116]:


all_responses


# In[121]:


classes[:-4]


# In[122]:


all_responses[all_responses['course_Id'].isin(classes[:-4])]


# In[111]:


for index in range(0,len(all_responses),1000):
    df = all_responses[index:index+1000]
    collection.insert_many(df.to_dict('records'))
    print(str(index)+" rows inserted.")


# In[161]:


dd=pd.DataFrame(list(db.aiquestionslight.find()))


# In[162]:


dd


# In[163]:


questions = get_classes(dd)


# In[165]:


questions


# In[164]:


questions_df = questions[questions['courseId'].isin(classes[6:12])]


# In[147]:


df['subjectId'].value_counts()


# In[148]:


df.to_csv("questions_df.csv")


# In[149]:


collections = db.aiquestionslight


# In[150]:


collections.insert_many(df.to_dict('records'))


# In[146]:


df = questions_df.groupby('courseId').sample(n=1000, random_state=42)


# In[143]:


df['courseId'].value_counts()


# In[135]:


classes[6:12]


# In[120]:


pd.DataFrame(list(db.studentresponses.find({"course_Id":'5fff7329de0bdb47f826feb0'})))


# In[83]:


collection.insert_many(all_responses.to_dict('records'))


# In[137]:


lesson_map[lesson_map['courseId'].isin(classes[6:12])]


# In[ ]:





# In[ ]:





# In[ ]:





# In[38]:


responses_df, le_difficulty, le_subject_name, le_userid,le_questionId = label_transforms(responses_df)


# In[39]:


responses_df


# In[40]:


# train model


# In[41]:


# test accuracy


# In[42]:


classification(responses_df)


# In[64]:


# deploy model locally


# In[65]:


questions_with_class


# In[ ]:




