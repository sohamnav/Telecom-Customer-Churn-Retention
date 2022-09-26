
# coding: utf-8
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics
from flask import Flask, request, render_template
import pickle
from sklearn.preprocessing import LabelEncoder

app = Flask("__name__")

df_1=pd.read_csv("appexcel.csv")

q = ""

@app.route("/")
def loadPage():
	return render_template('home2.html', query="")

@app.route("/", methods=['POST'])
def predict():
    
    inputQuery1 = request.form['query1']
    inputQuery2 = request.form['query2']
    inputQuery3 = request.form['query3']
    inputQuery4 = request.form['query4']
    inputQuery5 = request.form['query5']
    inputQuery6 = request.form['query6']
    inputQuery7 = request.form['query7']
    inputQuery8 = request.form['query8']
    inputQuery9 = request.form['query9']
    inputQuery10 = request.form['query10']
    inputQuery11 = request.form['query11']
    inputQuery12 = request.form['query12']
    inputQuery13 = request.form['query13']
    inputQuery14 = request.form['query14']
    inputQuery15 = request.form['query15']
    inputQuery16 = request.form['query16']
    inputQuery17 = request.form['query17']
    inputQuery18 = request.form['query18']
    inputQuery19 = request.form['query19']
    inputQuery20 = request.form['query20']
    inputQuery21 = request.form['query21']
    inputQuery22 = request.form['query22']
    inputQuery23 = request.form['query23']
    inputQuery24 = request.form['query24']
    inputQuery25 = request.form['query25']
    inputQuery26 = request.form['query26']
    inputQuery28 = request.form['query28']
    inputQuery29 = request.form['query29']
    inputQuery30 = request.form['query30']

    model = pickle.load(open("model.sav", "rb"))
   
    data = [[inputQuery1,inputQuery2,inputQuery3,inputQuery4,inputQuery5,inputQuery6,inputQuery7,inputQuery8,inputQuery9,inputQuery10,
            inputQuery11,inputQuery12,inputQuery13,inputQuery14,inputQuery15,inputQuery16,inputQuery17,inputQuery18,inputQuery19,
            inputQuery20,inputQuery21,inputQuery22,inputQuery23,inputQuery24,inputQuery25,inputQuery26,inputQuery28,
            inputQuery29,inputQuery30]] 
    
    new_df = pd.DataFrame(data, columns = ['gender','age','married','numberofdependents','numberofreferrals','tenureinmonths','offer','phoneservice',
                                           'avgmonthlylongdistancecharges','multiplelines','internetservice','internettype','avgmonthlygbdownload','onlinesecurity',
                                           'onlinebackup','deviceprotectionplan','premiumtechsupport','streamingtv','streamingmovies',
                                           'streamingmusic','unlimiteddata','contract','paperlessbilling','paymentmethod','monthlycharge',
                                           'totalcharges','totalextradatacharges','totallongdistancecharges','totalrevenue'])
  
    le = LabelEncoder()
    list1=['gender','married','phoneservice','internetservice','paperlessbilling']
    for col in list1:
        if df_1[col].dtype == 'object':
            if len(list(df_1[col].unique())) <= 2:
                le.fit(df_1[col])
                df_1[col] = le.transform(df_1[col])
        else:
            break

    new_df['gender'] = np.where(new_df.gender == 'Female',0,1)
    new_df['married'] = np.where(new_df.married == 'Yes',1,0)
    new_df['phoneservice'] = np.where(new_df.phoneservice == 'Yes',1,0)
    new_df['internetservice'] = np.where(new_df.internetservice == 'Yes',1,0)
    new_df['paperlessbilling'] = np.where(new_df.paperlessbilling == 'Yes',1,0)
    
    df_2 = pd.concat([df_1, new_df],ignore_index=True)
   
    #df_2.to_csv('labelappexcel.csv') 
 
    # Group the tenure in bins of 12 months
    labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
    
    df_2['tenure_group'] = pd.cut(df_2.tenureinmonths.astype(int), range(1, 80, 12), right=False, labels=labels)
    #drop column customerID and tenure
    df_2.drop(columns= ['tenureinmonths'], axis=1, inplace=True)   
    
    new_df__dummies = pd.get_dummies(df_2[['offer','multiplelines','internettype','onlinesecurity','onlinebackup','deviceprotectionplan',
                                           'premiumtechsupport','streamingtv','streamingmovies','streamingmusic','unlimiteddata','contract',                                
                                           'paymentmethod','tenure_group']])
 
    
    #new_df__dummies.to_csv('appexceldummies.csv')
    
    df_2.drop(columns= ['offer','multiplelines','internettype','onlinesecurity','onlinebackup','deviceprotectionplan',
                        'premiumtechsupport','streamingtv','streamingmovies','streamingmusic','unlimiteddata','contract',                                
                        'paymentmethod','tenure_group'], axis=1, inplace=True)  

    df_3=pd.concat([df_2,new_df__dummies],axis=1,ignore_index=False) 

    df_3 = df_3[df_3.columns.drop(list(df_3.filter(regex='Unnamed :')))]

    #df_3.to_csv('processedappexcel.csv')

    single = model.predict(df_3.tail(1))
    probablity = model.predict_proba(df_3.tail(1))[:,1]
    
    if single==1:
        o1 = "This customer is likely to be churn"
        o2=np.round(probablity*100,2)
        o3 = "{} %".format(o2)

    else:
        o1 = "This customer is likely to Stay"
        o2=np.round(probablity*100,2)
        o3 = "{} %".format(o2)

    return render_template('home2.html', output1=o1, output2=o3, 
                           query1 = request.form['query1'], 
                           query2 = request.form['query2'],
                           query3 = request.form['query3'],
                           query4 = request.form['query4'],
                           query5 = request.form['query5'], 
                           query6 = request.form['query6'], 
                           query7 = request.form['query7'], 
                           query8 = request.form['query8'], 
                           query9 = request.form['query9'], 
                           query10 = request.form['query10'], 
                           query11 = request.form['query11'], 
                           query12 = request.form['query12'], 
                           query13 = request.form['query13'], 
                           query14 = request.form['query14'], 
                           query15 = request.form['query15'], 
                           query16 = request.form['query16'], 
                           query17 = request.form['query17'],
                           query18 = request.form['query18'], 
                           query19 = request.form['query19'],
                           query20 = request.form['query20'],
                           query21 = request.form['query21'],
                           query22 = request.form['query22'],
                           query23 = request.form['query23'],
                           query24 = request.form['query24'],
                           query25 = request.form['query25'],
                           query26 = request.form['query26'],
                           query28 = request.form['query28'],
                           query29 = request.form['query29'],
                           query30 = request.form['query30'])


app.run(debug=True)
