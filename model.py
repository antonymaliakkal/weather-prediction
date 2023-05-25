import pandas as pd
import numpy as np
import pickle

data = pd.read_csv('dataset.csv')

#label encoding the output column

from sklearn.preprocessing import LabelEncoder
l = LabelEncoder()
data['weather'] = l.fit_transform(data['weather'])
label_values = {0 : 'drizzle' , 1 : 'fog' , 2 : 'rain' , 3 : 'snow' , 4 : 'sun'}

#outlier detection

outlier_indices = []
columns = ['precipitation','temp_max','temp_min','wind']
for c in columns:
    Q1 = np.percentile(data[c],25)
    Q3 = np.percentile(data[c],75)
    IQR = Q3 - Q1
    outlier_step = IQR * 1.5
    outlier_list_col = data[(data[c] < Q1 - outlier_step) | (data[c] > Q3 + outlier_step)].index
    outlier_indices.extend(outlier_list_col)
        
multiple_outliers = []
for i in outlier_indices:
    x = 0
    for j in outlier_indices:
        if i == j:
            x = x + 1
    if x>1:
        multiple_outliers.append(i)

#deleting the outliers

data = data.drop(multiple_outliers,axis=0).reset_index(drop=True)

#creating a data frame

df = pd.DataFrame(data)
df.drop('date',inplace=True,axis=1)
y = df['weather']
x = df.drop('weather',axis=1)

#pre-processing the data

from sklearn.preprocessing import StandardScaler
s = StandardScaler()
x_scaled = s.fit_transform(x)

#spliting the dataset

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x_scaled,y,test_size=0.25)

#training the random forest model

from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train,y_train)

#saving the model with pickle

pickle.dump(rf_clf, open('random_forest.pkl', 'wb'))

#loading the saved model

model = pickle.load(open('random_forest.pkl','rb'))
print(model.predict([[0.0,12.8,6.1,4.3]]))

#training the knn model

from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train,y_train)

#saving the model with pickle

pickle.dump(rf_clf, open('knn.pkl', 'wb'))

#loading the saved model

model = pickle.load(open('knn.pkl','rb'))
print(model.predict([[0.0,12.8,6.1,4.3]]))  

#training the ann model

import tensorflow as tf
from tensorflow import keras
ann = keras.Sequential([
    keras.layers.Dense(128 , activation = 'relu' , input_shape = (4,)),
    keras.layers.Dense(64 , activation = 'relu'),
    keras.layers.Dense(5 ,  activation = 'softmax')
])
ann.compile(optimizer = 'adam' , loss = 'sparse_categorical_crossentropy' , metrics = ['accuracy'])
ann.fit(X_train, y_train, epochs=80, batch_size=32, validation_data=(X_test, y_test))

#saving the model with pickle

pickle.dump(rf_clf, open('ann.pkl', 'wb'))

#loading the saved model

model = pickle.load(open('ann.pkl','rb'))
print(model.predict([[0.0,12.8,6.1,4.3]])) 

#traing ann with sklearn

from sklearn.neural_network import MLPClassifier
ann = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', 
                      alpha=0.001, batch_size=32, learning_rate='constant', 
                      learning_rate_init=0.001, max_iter=200, random_state=42)
ann.fit(X_train, y_train)

#saving the model with pickle

pickle.dump(rf_clf, open('ann1.pkl', 'wb'))

#loading the saved model

model = pickle.load(open('ann1.pkl','rb'))
print(model.predict([[0.0,12.8,6.1,4.3]])) 

#training gbc model

from sklearn.ensemble import GradientBoostingClassifier
gbc_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
gbc_clf.fit(X_train,y_train)

#saving the model with pickle

pickle.dump(rf_clf, open('gbc.pkl', 'wb'))

#loading the saved model

model = pickle.load(open('gbc.pkl','rb'))
print(model.predict([[0.0,12.8,6.1,4.3]])) 