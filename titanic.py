import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import base64
from io import BytesIO

pd.set_option("display.precision", 3)
#DATA ACQUISITION
df = pd.read_csv('train.csv')
df = df.round({'Fare': 1, 'Age':0})
df.fillna(0,inplace=True)
df['Age'] = df['Age'].astype(int)
df_original = df

#DATA TREATMENT
df = df.drop(['Name','Ticket','Cabin','Embarked'], axis=1)
df['Sex_F'] = np.where(df['Sex'] == 'female', 1, 0)
df['Pclass_1'] = np.where(df['Pclass'] == 1, 1, 0)
df['Pclass_2'] = np.where(df['Pclass'] == 2, 1, 0)
df['Pclass_3'] = np.where(df['Pclass'] == 3, 1, 0)
df = df.drop(['Pclass','Sex'], axis = 1)

#DATA SAMPLING AND MODEL TRAINING(30% to test, 70% to train)
xTrain, xTest, yTrain, yTest = train_test_split(df.drop(['Survived'], axis=1), 
                                                df['Survived'],
                                                test_size = 0.3,
                                                random_state = 1234)


rndForest = RandomForestClassifier(n_estimators = 1000,
                                    criterion = 'gini',
                                    max_depth = 5)
rndForest.fit(xTrain, yTrain)

#DATA ASSESSMENT
probability = rndForest.predict_proba(df.drop('Survived', axis=1))[:,1]
classification = rndForest.predict(df.drop('Survived',axis=1))

df.insert(loc=2, column="Survived (Model)", value= classification)
df.insert(loc=3, column="Survive probability", value= probability)
testIds = list(xTest.loc[:,'PassengerId'])
df.loc[df['PassengerId'].isin(testIds), 'Is Test Set?'] = 1
df.fillna(0,inplace=True)
df['Is Test Set?'] = df['Is Test Set?'].astype(int)

df_result = df
#PRESENTATION FILTERS
df_original = df_original.set_index(['PassengerId'])
df_result = df_result.set_index(['PassengerId'])

df_original['Cabin'] = df_original['Cabin'].replace(0,np.nan)
df_original.to_html("data_original.html")
df_result.to_html("data_result.html")



#PLOTS
sProb = list(df_result.iloc[:,2])
pId = range(len(sProb))
col = []
lab = []
for i in range(len(pId)):
    if list(df_result.iloc[:,-1])[i] == 1:
        col.append('magenta')
        lab.append('Test Set')
    else:
        col.append('blue')
        lab.append('Train Set')

plot1 = plt.figure(1)
dead=[]
alive=[]
for i in range(len(pId)):
    if list(df_result.iloc[:,0])[i] == 0:
        dead.append([i,sProb[i]])
    else:
        alive.append([i,sProb[i]])

plt.subplot(1, 2, 1) #the dead
plt.subplot(1, 2, 1).title.set_text('Passengers that Died\nPredicted chance of survival')
for d in range(len(dead)):
    plt.scatter(dead[d][0], dead[d][1], c = col[dead[d][0]])

plt.subplot(1, 2, 2) #the alive
plt.subplot(1, 2, 2).title.set_text('Passengers that Survived\nPredicted chance of survival')

oneTrain = True
oneTest = True
for a in range(len(alive)):
    if (lab[alive[a][0]] == 'Test Set' and oneTest):
        oneTest = False
        plt.scatter(alive[a][0], alive[a][1], c = col[alive[a][0]], label = lab[alive[a][0]])
    elif (lab[alive[a][0]] == 'Train Set' and oneTrain):
        oneTrain = False
        plt.scatter(alive[a][0], alive[a][1], c = col[alive[a][0]], label = lab[alive[a][0]])
    else:
        plt.scatter(alive[a][0], alive[a][1], c = col[alive[a][0]])
plt.legend()

plot2 = plt.figure(2)
xCounted = ['Surv.\nTestSet', 'Predicted\nSurv. TestSet', 'Surv.\nTotal', 'Predicted\nSurv. Total']
yCounted = [np.dot( list(df_result.iloc[:,0]), list(df_result.iloc[:,-1]) ), np.dot( list(df_result.iloc[:,1]), list(df_result.iloc[:,-1]) ), 
            sum(list(df_result.iloc[:,0])), sum(list(df_result.iloc[:,1]))]

            # sum(np.dot( list(df_result.iloc[:,1]), list(df_result.iloc[:,-1]) ))
plt.bar(xCounted, yCounted)

overrall_result = pd.DataFrame({'Perc. Error TestSet': ["{0:.1f}%".format( ((yCounted[1]-yCounted[0])/(yCounted[1])) * 100 )],
                                'Perc. Error Global': ["{0:.1f}%".format( ((yCounted[3]-yCounted[2])/(yCounted[3])) * 100 )] })
overrall_result.to_html("overrall_result.html", index = False)

import os
#CREATE DATA RESULT HTML FILE
with open("data_result.html") as tableDataResult:
    tableDataResult = tableDataResult.read()
    tableDataResult = tableDataResult.replace('<table border="1" class="dataframe">', '<h2>Table 1 - RandomForest results</h2>\n<table class="data_result">')
with open("data_result.html", "w") as file_to_write:
  	file_to_write.write(tableDataResult)
del file_to_write
os.remove("data_result.html")

#CREATE DATA ORIGINAL HTML FILE
with open("data_original.html") as tableDataOriginal:
    tableDataOriginal = tableDataOriginal.read()
    tableDataOriginal = tableDataOriginal.replace('<table border="1" class="dataframe">', '<h2>Table 2 - Original data (from Kaggle)</h2>\n<table class="data_original">')
with open("data_original.html", "w") as file_to_write:
    file_to_write.write(tableDataOriginal)
del file_to_write
os.remove("data_original.html")

with open("overrall_result.html") as tableOverrall:
    tableOverrall = tableOverrall.read()
    tableOverrall = tableOverrall.replace('<table border="1" class="dataframe">', '</div>\n<div class="right-container">\n<h2>Table 3 - Summary of results</h2>\n<table class="overrall_result">')
with open("overrall_result.html", "w") as file_to_write:
    file_to_write.write(tableOverrall)
del file_to_write
os.remove("overrall_result.html")


with open("main.html") as mainHtml:
    mainHtml = mainHtml.read()

with open("end.html") as endHtml:
    endHtml = endHtml.read()

tmpfile = BytesIO()
plot1.savefig(tmpfile, format='png')
encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
htmlPlot1 = '<h2>Figure 1 - Chance of surviving (left=dead) (right=survived)</h2>\n<img class="plot1" src=\'data:image/png;base64,{}\'>'.format(encoded)

tmpfile = BytesIO()
plot2.savefig(tmpfile, format='png')
encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
htmlPlot2 = '<h2>Figure 2 - Comparison of results with predictions</h2>\n<img class="plot1" src=\'data:image/png;base64,{}\'>'.format(encoded)


with open("index.html", "w") as file_to_write:
  	file_to_write.write(mainHtml+tableDataResult+tableDataOriginal+tableOverrall+htmlPlot1+htmlPlot2+endHtml)
del file_to_write

os.startfile("index.html")
