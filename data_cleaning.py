import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re


data = pd.read_excel('SATA_CompiledData.xlsx',sheet_name='Compiled')

pd.set_option('display.max_columns',None) #Showing all columns
pd.set_option("display.max_rows", None)





data['Iron Saturation (%)'] = data['Iron Saturation (%)'].replace(np.nan, 0)
data = data.replace('Nil', 'No')

data['Ferritin (ug/L)'] = data['Ferritin (ug/L)'].replace('<0.5', '0.5')



## Cleaning Age Column

data['Age'] = data['Age'].replace('Nil', np.nan)
data['Age'] = data['Age'].replace('Yes', np.nan)
data['Age'] = data['Age'].replace('No', np.nan)
data['Age'] = data['Age'].replace('-', np.nan)
data['Age'] = data['Age'].replace(np.nan, data['Age'].mean())
data['Age'] = data['Age'].replace(0, data['Age'].mean())
data['Age'] = data['Age'].astype(np.int64)


## Cleaning height Column
data['Height\n(metres)'] = data['Height\n(metres)'].replace('Nil', np.nan)
data['Height\n(metres)'] = data['Height\n(metres)'].replace('No', np.nan)
data['Height\n(metres)'] = data['Height\n(metres)'].replace('-', np.nan)
data['Height\n(metres)'] = data['Height\n(metres)'].replace('5 inch',np.nan)
data['Height\n(metres)'] = data['Height\n(metres)'].replace('5\'3"',1.6)
data['Height\n(metres)'] = data['Height\n(metres)'].replace('5\'1"',1.54)
data['Height\n(metres)'] = data['Height\n(metres)'].replace('''5'6"''',1.67)
data['Height\n(metres)'] = data['Height\n(metres)'].replace(np.nan, data['Height\n(metres)'].mean())
data['Height\n(metres)'] = data['Height\n(metres)'].replace(0, data['Height\n(metres)'].mean())
data['Height\n(metres)'] = data['Height\n(metres)'].astype(np.float64)



## Cleaninig Weight Column
data['Weight\n(kg)'] = data['Weight\n(kg)'].replace('-', np.nan)
data['Weight\n(kg)'] = data['Weight\n(kg)'].replace('No', np.nan)

#data['Weight\n(kg)'] = data['Weight\n(kg)'].astype(np.float64)
data['Weight\n(kg)'] = data['Weight\n(kg)'].replace(874, data['Weight\n(kg)'].mean())

data['Weight\n(kg)'] = data['Weight\n(kg)'].replace(np.nan, data['Weight\n(kg)'].mean())
data['Weight\n(kg)'] = data['Weight\n(kg)'].replace(0, data['Weight\n(kg)'].mean())
data['Weight\n(kg)'] = data['Weight\n(kg)'].astype(np.float64)





## Cleaninig Premeno-pausal column
data['Premenopausal'] = data['Premenopausal'].replace('No answer','No')
data['Premenopausal'] = data['Premenopausal'].replace(' Yes','Yes')
data['Premenopausal'] = data['Premenopausal'].replace('N.A.','No')
data['Premenopausal'] = data['Premenopausal'].replace('Ni','No')
data['Premenopausal'] = data['Premenopausal'].replace(0,'No')
data['Premenopausal'] = data['Premenopausal'].replace(np.nan,'No')

## Cleaning Period Column
data['Periods (female only)'] = data['Periods (female only)'].replace('nil',0)
data['Periods (female only)'] = data['Periods (female only)'].replace('No answer','No')
data['Periods (female only)'] = data['Periods (female only)'].replace('N.A.','No')
data['Periods (female only)'] = data['Periods (female only)'].replace('No',0)
data['Periods (female only)'] = data['Periods (female only)'].replace(' ',0)
data['Periods (female only)'] = data['Periods (female only)'].replace("Few", np.nan)
data['Periods (female only)'] = data['Periods (female only)'].replace("Irregular", np.nan)

data['Periods (female only)'] = data['Periods (female only)'].replace("None", 0)
data['Periods (female only)'] = data['Periods (female only)'].replace("7 to 8", np.nan)

data['Periods (female only)'] = data['Periods (female only)'].replace('0 (pregnant)',0)

data['Weight\n(kg)'] = data['Weight\n(kg)'].astype(np.int64)

data['Periods (female only)'] = data['Periods (female only)'].replace(np.nan, data['Periods (female only)'].mean())


## Cleaning Competition Level

data['Competition Level (1-6)'] = data['Competition Level (1-6)'].replace(['No answer','No','Nil','Nil '],0)
data['Competition Level (1-6)'] = data['Competition Level (1-6)'].replace(['2,3','1,2','3,4','3,5','2,3,4','2, 3','1,3'],3)
data['Competition Level (1-6)'] = data['Competition Level (1-6)'].replace(['2 & 3','1 & 2','2 & 4','1 to 6','1 & 3','1&2','3 & 4','2.3','2.3 '],3)
data['Competition Level (1-6)'] = data['Competition Level (1-6)'].replace('1,2,3,4,5 & 6',0)
data['Competition Level (1-6)'] = data['Competition Level (1-6)'].replace('3,4,5',4)
data['Competition Level (1-6)'] = data['Competition Level (1-6)'].replace(np.nan,0)

data['Competition Level (1-6)'] = data['Competition Level (1-6)'].replace([12.0,2.3],0)




## Cleaning Period Advice Column

data['Period Advice\n(female only)'] = data['Period Advice\n(female only)'].replace(['no',' No',' ',0],'No')
data['Period Advice\n(female only)'] = data['Period Advice\n(female only)'].replace(['DK','Dk','N.A.'],'DK')
data['Period Advice\n(female only)'] = data['Period Advice\n(female only)'].replace(np.nan,'DK')


#Restructing Gender Column
data['Gender'] = data['Gender'].replace({"F":1,"M":0})


# Cleaning Anaemia Column
data['Anaemia'] = data['Anaemia'].replace(['no',' No',' ',0],'No')
data['Anaemia'] = data['Anaemia'].replace(['DK','Dk'],'DK')
data['Anaemia'] = data['Anaemia'].replace(np.nan,'DK')



##Cleaning Anaemia Last 3 months
data['Anaemia last 3 months'] = data['Anaemia last 3 months'].replace(['no',' No',' ',0],'No')
data['Anaemia last 3 months'] = data['Anaemia last 3 months'].replace(['DK','Dk','NIl','Nill'],'DK')
data['Anaemia last 3 months'] = data['Anaemia last 3 months'].replace(np.nan,'DK')


## Clenaing iron deficiency

data['Iron Deficient'] = data['Iron Deficient'].replace(['no',' No',' ',' No',0],'No')
data['Iron Deficient'] = data['Iron Deficient'].replace(['DK','Dk','NDK','NDk'],'DK')
data['Iron Deficient'] = data['Iron Deficient'].replace(np.nan,'DK')



#Cleaning Iron def last 3 months
data['Iron Deficient last 3 months'] = data['Iron Deficient last 3 months'].replace(['no',' No',' ',' No','nil',' Nil','Niil','NIl',0],'No')
data['Iron Deficient last 3 months'] = data['Iron Deficient last 3 months'].replace(['DK','Dk','NDK','NDk'],'DK')
data['Iron Deficient last 3 months'] = data['Iron Deficient last 3 months'].replace(np.nan,'DK')


##Cleaning Iron sat
data['Iron Saturation (%)'].value_counts()
data['Iron Saturation (%)'] = data['Iron Saturation (%)'].replace('*5',5)
data['Iron Saturation (%)'] = data['Iron Saturation (%)'].replace('*11',11)
data['Iron Saturation (%)'] = data['Iron Saturation (%)'].astype(np.int64)



## . Cleaning iron Suppl
data['Iron Supplement'] = data['Iron Supplement'].replace(['no',' No',' ',' No',0],'No')
data['Iron Supplement'] = data['Iron Supplement'].replace(['DK','Dk','NDK','NDk'],'DK')
data['Iron Supplement'] = data['Iron Supplement'].replace('Yes, during pregnancy', 'Yes')
data['Iron Supplement'] = data['Iron Supplement'].replace(np.nan,'DK')

#Cleaning Iron Supl last 3 months
data['Iron Supplement last 3 months'] = data['Iron Supplement last 3 months'].replace(['no',' No',' ',' No',' Nil',0],'No')
data['Iron Supplement last 3 months'] = data['Iron Supplement last 3 months'].replace(['DK','Dk','NDK','NDk'],'DK')
data['Iron Supplement last 3 months'] = data['Iron Supplement last 3 months'].replace('Yes, during pregnancy', 'Yes')
data['Iron Supplement last 3 months'] = data['Iron Supplement last 3 months'].replace(np.nan,'DK')


data['Heavy Menstrual Bleeding_Imp'] = data['Heavy Menstrual Bleeding_Imp'].replace(['N.A.',0,4],'No')

data['Heavy Menstrual Bleeding \n1'] = data['Heavy Menstrual Bleeding \n1'].replace(['no',' No','N.A.',' ',0,2,'NI','Ni','Nii'],'No')
data['Heavy Menstrual Bleeding \n1'] = data['Heavy Menstrual Bleeding \n1'].replace(['DK','Dk'],'DK')
data['Heavy Menstrual Bleeding \n1'] = data['Heavy Menstrual Bleeding \n1'].replace('Yes ','Yes')
data['Heavy Menstrual Bleeding \n1'] = data['Heavy Menstrual Bleeding \n1'].replace(np.nan,'Yes')

data['Heavy Menstrual Bleeding_Imp'] = data['Heavy Menstrual Bleeding_Imp'].replace([np.nan,4,0],'No')


#data['Heavy Menstrual Bleeding \n2'].value_counts()

data['Heavy Menstrual Bleeding \n2'] = data['Heavy Menstrual Bleeding \n2'].replace(['no',' No',' ','N.A.',0,2,4,'NI','Ni','Nii'],'No')
data['Heavy Menstrual Bleeding \n2'] = data['Heavy Menstrual Bleeding \n2'].replace(['DK','Dk'],'DK')
data['Heavy Menstrual Bleeding \n2'] = data['Heavy Menstrual Bleeding \n2'].replace('Yes ','Yes')
data['Heavy Menstrual Bleeding \n2'] = data['Heavy Menstrual Bleeding \n2'].replace(np.nan,'Yes')


#data['Heavy Menstrual Bleeding \n3'].value_counts()

data['Heavy Menstrual Bleeding \n3'] = data['Heavy Menstrual Bleeding \n3'].replace(['no',' No',' ',0,2,4,'NI','N.A.','Ni','Nii'],'No')
data['Heavy Menstrual Bleeding \n3'] = data['Heavy Menstrual Bleeding \n3'].replace(['DK','Dk'],'DK')
data['Heavy Menstrual Bleeding \n3'] = data['Heavy Menstrual Bleeding \n3'].replace('Yes ','Yes')
data['Heavy Menstrual Bleeding \n3'] = data['Heavy Menstrual Bleeding \n3'].replace(np.nan,'Yes')


#data['Heavy Menstrual Bleeding \n4'].value_counts()

data['Heavy Menstrual Bleeding \n4'] = data['Heavy Menstrual Bleeding \n4'].replace(['no',' No',' ',0,2,4,'NI','Ni','Nii','N.A.'],'No')
data['Heavy Menstrual Bleeding \n4'] = data['Heavy Menstrual Bleeding \n4'].replace(['DK','Dk'],'DK')
data['Heavy Menstrual Bleeding \n4'] = data['Heavy Menstrual Bleeding \n4'].replace('Yes ','Yes')
data['Heavy Menstrual Bleeding \n4'] = data['Heavy Menstrual Bleeding \n4'].replace(np.nan,'Yes')


#data['HMB? (Yes/No)'].value_counts()

data['HMB? (Yes/No)'] = data['HMB? (Yes/No)'].replace('yes','Yes')
data['HMB? (Yes/No)'] = data['HMB? (Yes/No)'].replace(np.nan,'No')


#data.to_csv('trial.csv',index=False)

#data.to_excel('temp.xlsx',index=False)

data['MFI_1'] = data['MFI_1'].replace(np.nan,'No')
data['MFI_1'] = data['MFI_1'].replace('No',0)
data['MFI_1'] = data['MFI_1'].astype(np.float64)

def MFs(df):
    a = df.copy()
    a.loc[:] = a.loc[:].replace(np.nan,'No')
    a.loc[:] = a.loc[:].replace('No',1)
    a.loc[:] = a.loc[:].replace(0,1)
    a.loc[:] = a.loc[:].astype(np.int64)
    return a



#cols = ['MFI_1','MFI_5','MFI_12','MFI_16']




data['MFI_1'] = MFs(data.iloc[:,-20])
data['MFI_2'] = MFs(data.iloc[:,-19])
data['MFI_3'] = MFs(data.iloc[:,-18])
data['MFI_4'] = MFs(data.iloc[:,-17])
#data['MFI_5'] = MFs(data.iloc[:,-16])
data['MFI_6'] = MFs(data.iloc[:,-15])
data['MFI_7'] = MFs(data.iloc[:,-14])
data['MFI_8'] = MFs(data.iloc[:,-13])
data['MFI_9'] = MFs(data.iloc[:,-12])
data['MFI_10'] = MFs(data.iloc[:,-11])
#data['MFI_11'] = MFs(data.iloc[:,-10])
data['MFI_12'] = MFs(data.iloc[:,-9])
data['MFI_13'] = MFs(data.iloc[:,-8])
data['MFI_14'] = MFs(data.iloc[:,-7])
data['MFI_15'] = MFs(data.iloc[:,-6])
data['MFI_16'] = MFs(data.iloc[:,-5])
data['MFI_17'] = MFs(data.iloc[:,-4])
data['MFI_18'] = MFs(data.iloc[:,-3])
data['MFI_19'] = MFs(data.iloc[:,-2])
data['MFI_20'] = MFs(data.iloc[:,-1])


data['MFI_5'] = data['MFI_5'].replace(['No','1 & 4','2 & 4',0,np.nan],1)
data['MFI_5'] = data['MFI_5'].astype(np.int64)
data['MFI_5'].value_counts()

data['MFI_11'] = data['MFI_11'].replace(['No','43',43,np.nan],0)
data['MFI_11'] = data['MFI_11'].astype(np.int64)

def reverse_order(df):
    a = df.copy()
    a.iloc[:] = a.iloc[:].replace(5,'1')
    a.iloc[:] = a.iloc[:].replace(4,'2')
    a.iloc[:] = a.iloc[:].replace(3,'3')
    a.iloc[:] = a.iloc[:].replace(2,'4')
    a.iloc[:] = a.iloc[:].replace(1,'5')
    a.loc[:] = a.loc[:].astype(np.float64)
    return a

data['MFI_5'] = reverse_order(data['MFI_5'])
data['MFI_16'] = reverse_order(data['MFI_16'])



data['MFI_2'] = reverse_order(data['MFI_2'])
data['MFI_9'] = reverse_order(data['MFI_9'])
data['MFI_10'] = reverse_order(data['MFI_10'])
data['MFI_13'] = reverse_order(data['MFI_13'])
data['MFI_14'] = reverse_order(data['MFI_14'])
data['MFI_17'] = reverse_order(data['MFI_17'])
data['MFI_18'] = reverse_order(data['MFI_18'])
data['MFI_19'] = reverse_order(data['MFI_19'])
#data['MFI_16'] = reverse_order(data['MFI_16'])
#data['MFI_16'] = reverse_order(data['MFI_16'])

def MFshhh(df):
    a = df.copy()
    a.loc[:] = a.loc[:].replace(['No', 'More than 300','180-300','5 to 10'],0)
    a.loc[:] = a.loc[:].replace(['30-90','60-90','60 to 90','60-120','30-60','3x60','60 to 90 ','3x60 ','30 - 60'],60)
    a.loc[:] = a.loc[:].replace('1x45',45)
    a.loc[:] = a.loc[:].replace(np.nan, a.loc[:].mean())

    a.loc[:] = a.loc[:].astype(np.float64)
    return a

data['Cycling\nTime (minutes)'] = MFshhh(data['Cycling\nTime (minutes)'])
data['Swimming\nTime (minutes)'] = MFshhh(data['Swimming\nTime (minutes)'])
data['Running\nTime (minutes)'] = MFshhh(data['Running\nTime (minutes)'])


data = data.drop('Other\nTime (minutes)', 1)


def fatigue(df):
    a = df.copy()
    a['General_fat'] = (a['MFI_1'] + a['MFI_5'] + a['MFI_12'] + a['MFI_16'])//4
    a['Physical_fat'] = (a['MFI_2'] + a['MFI_8'] + a['MFI_14'] + a['MFI_20'])//4
    a['Mental_fat'] = (a['MFI_4'] + a['MFI_9'] + a['MFI_15'] + a['MFI_18'])//4
    a['Reduced_fat'] = (a['MFI_3'] + a['MFI_6'] + a['MFI_10'] + a['MFI_17'])//4
    a['Reduced_activity'] = (a['MFI_7'] + a['MFI_11'] + a['MFI_13'] + a['MFI_19'])//4


    
    return a

data = fatigue(data)

data.to_excel('Clean_data.xlsx',index=False)








######  Model Implementation ##################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC,SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier


from sklearn.metrics import mean_squared_error,f1_score


from sklearn.svm import SVR,LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

df = pd.read_excel('Clean_data.xlsx')

pd.set_option('display.max_columns',None)
pd.set_option("display.max_rows", None)


'''def f(row):
    if row['Ferritin (ug/L)'] > 15 or row['Ferritin (ug/L)'] < 30:
        val = 1
    elif row['Ferritin (ug/L)'] < 15:
        val = 0
    else:
        val = -1
    return val

df['IS'] = df['Ferritin (ug/L)'].apply(f,axis=1)'''

# create a list of our conditions
conditions = [
    (df['Ferritin (ug/L)'] < 16),
    (df['Ferritin (ug/L)'] > 15) & (df['Ferritin (ug/L)'] < 30),
    (df['Ferritin (ug/L)'] > 30),
    ]

values = ['High Risk', 'Medium Risk','Low Risk']

df['Target'] = np.select(conditions, values)

'''
df['Target'] = np.where(
    df['Ferritin (ug/L)'] < 15, 0, np.where(
    df['Ferritin (ug/L)'] >  15 , 1, -1)) '''



num_cols = df.select_dtypes(['int64','float64']).columns
num = df[num_cols]
#df['race_label'] = df.apply (lambda row: label_race(row), axis=1)

for cols in df.columns.drop(num_cols):
    df[cols] = df[cols].replace({"Yes":1,"No":0,"DK":2})


y = df['Target']
X = df.drop('Target',axis=1)
    
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size = 0.8,random_state = 2)
    
    
scaler = StandardScaler()
scaler.fit(X_train)



models={
    "LogisticRegression" : LogisticRegression(),
    "KNeighborsClassifier" : KNeighborsClassifier(),
    "DecisionTreeClassifier" : DecisionTreeClassifier(),
    "LinearSVC" : LinearSVC(),
    "SVC" : SVC(),
    "MLPClassifier" : MLPClassifier(),
    "RandomForestClassifier" : RandomForestClassifier(),
    "GradientBoostingClassifier" : GradientBoostingClassifier(),
}

for name,model in models.items():
    model.fit(X_train,y_train)
    print(name + "Model Trained.")
    

## Results
for name,model in models.items():
    print(name + ': {:.2f}%'.format(model.score(X_test,y_test) * 100))

for name,model in models.items():
    print(name + ': {:.2f}%'.format(model.score(X_train,y_train) * 100))


### Regression ####

y = df['Ferritin (ug/L)']
X = df.drop(['Target','Ferritin (ug/L)'],axis=1)

X_train,X_test,y_train,y_test = train_test_split(X,y,train_size = 0.8,random_state = 2)
    
    
scaler = StandardScaler()
scaler.fit(X_train)

reg_models = {
    "Linear Regression " : LinearRegression(),
    "DecisionTreeRegression" : DecisionTreeRegressor(),
    "LinearSVR" : LinearSVR(),
    "SVR" : SVR(),
    "MLPRegressor" : MLPRegressor(),
    "RandomForestRegressor" : RandomForestRegressor(),
    "GradientBoostingRegressor" : GradientBoostingRegressor(),
}

for name,model in reg_models.items():
    model.fit(X_train,y_train)
    print(name + "Model Trained.")


for name,model in reg_models.items():
    y_pred = model.predict(X_train)
    #((model.predict(X_test)))
    pred = mean_squared_error(y_train, y_pred)
    print(name + 'MSE Score : ', pred)


df.drop(df.iloc[:, -26:], inplace = True, axis = 1)
df.drop(df.iloc[:,:2],inplace = True,axis = 1)


y = df['Ferritin (ug/L)']
#df['Height\n(metres)']
#df['Weight\n(kg)']
df['HB (g/dL)']
X = df.drop(['Weight\n(kg)','Height\n(metres)','Swimming\nTime (minutes)','Ferritin (ug/L)','Competition Level (1-6)','Running\nTime (minutes)','Cycling\nTime (minutes)'],axis=1)



X_train,X_test,y_train,y_test = train_test_split(X,y,train_size = 0.8,random_state = 2)
    
    
scaler = StandardScaler()
scaler.fit(X_train)


reg_models = {
    "Linear Regression " : LinearRegression(),
    "DecisionTreeRegression" : DecisionTreeRegressor(),
    "LinearSVR" : LinearSVR(),
    "SVR" : SVR(),
    "MLPRegressor" : MLPRegressor(),
    "RandomForestRegressor" : RandomForestRegressor(),
    "GradientBoostingRegressor" : GradientBoostingRegressor(),
}


for name,model in reg_models.items():
    model.fit(X_train,y_train)
    print(name + "Model Trained.")


for name,model in reg_models.items():
    y_pred = model.predict(X_train)
    #((model.predict(X_test)))
    pred = mean_squared_error(y_train, y_pred)
    print(name + 'MSE Score : ', pred)
