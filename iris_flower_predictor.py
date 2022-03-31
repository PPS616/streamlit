import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split

#Streamlit Styling
st.set_page_config(layout="wide")

st.title('Iris Flower Species Prediction App')

# Read Data
iris_df = pd.read_csv('iris-species.csv')

# Mapping Species Column
iris_df['Species'] = iris_df['Species'].map( {'Iris-setosa':0 , 'Iris-virginica':1 , 'Iris-versicolor':2 } )

#Splitting the Data
X = iris_df[ iris_df.columns[1:-1] ]
y = iris_df['Species']

X_train , X_test , y_train , y_test = train_test_split( X , y , test_size = 0.30 , random_state = 42)

#Model Training
from sklearn.svm import SVC

svc_model = SVC( kernel='linear' )
svc_model.fit( X_train , y_train )
score = svc_model.score( X_train , y_train )

#Streamlit Functions
def pred( s_len , s_wid , p_len , p_wid ):
    labels = svc_model.predict( [ [s_len , s_wid , p_len , p_wid] ] )
    labels = labels[0]
    if labels == 0:
        return 'Iris-setosa'
    elif labels == 1:
        return 'Iris-virginica'
    else:
        return 'Iris-versicolor'

#Streamlit Variables
s_len = st.slider('Select Sepal Length?', 0.0, 10.0 , key = '1')

s_wid = st.slider('Select Sepal Width?', 0.0, 10.0 , key = '2')

p_len = st.slider('Select Petal Length?', 0.0, 10.0 , key = '4')

p_wid = st.slider('Select Sepal Width?', 0.0, 10.0 , key = '5')

#Streamlit Button
if st.button('PREDICT'):
    predicted_label = pred( s_len , s_wid , p_len , p_wid )
    st.write(f'Model predicted : {predicted_label}')
    st.write(f'Model Score : {round( score , 2 ) }')
