#importing the Necessary Library
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.ensemble import RUSBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import gradio as gr


#Loading dataset
df = pd.read_csv('Churn Dataset.csv')
df.head()


#Converting Gender into int
df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})

x = df.drop(['Churn', 'ID'], axis=1)
y = df['Churn']

#Splitting the dataset into training and testing set 
x_train, x_test, y_train,y_test = train_test_split(x, y, test_size=0.33, random_state=46)

#Training Model With Random Classifier
model = RandomForestClassifier(random_state=46)
model.fit(x_train, y_train)

#Defining churn Predition Function
def predict_churn(Gender	,Age	,MembershipYears	,AvgMonthlySpend	,NumVisitsPerMonth,	UsedCoupons	,Churn):
    gender_num = 0 if Gender.lower() == 'Female' else 1
    Coupons_num = 1 if UsedCoupons.lower() == 'yes' else 0
    features = [[Age, gender_num, MembershipYears, AvgMonthlySpend ,Coupons_num, NumVisitsPerMonth]]
    pred = model.predict(features)
    return 'Churn' if pred == 1 else 'No churn'


#Gradio Interface
iface = gr.Interface(
        fn=predict_churn,
    inputs=[
        gr.Radio(['Female', 'Male'], label='Gender'),
        gr.Number(label='Age'),
        gr.Number(label='Membership Years'),
        gr.Number(label= 'Average Monthly Spending'),
        gr.Number(label='Number of visits per months'),
        gr.Radio(['yes', 'No'], label='Used Coupons')
    ],

    outputs='text',
    title='Churn Prediction'
)

iface.launch()