import streamlit as st
import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split




# sidebar
st.sidebar.header("Build model DNN")

file=st.sidebar.file_uploader("Select file")
activation_input=st.sidebar.selectbox("activation input",("relu","sigmoid","softmax"))

hiddenlayer=st.sidebar.slider("Hidden layer",1,5)
acc=[]
node=[]
for i in range(0,hiddenlayer):
    a=st.sidebar.number_input("hidden layer "+str(i+1),step=8,value=8)
    node.append(a)
    b=st.sidebar.selectbox("activation "+str(i+1),("relu","sigmoid","softmax"))
    acc.append(b)
activation_output=st.sidebar.selectbox("activation output",("relu","sigmoid","softmax"))
# LearningRate=st.sidebar.number_input("Learning rate",min_value=0.001,
#     step=0.001,
#     value=0.001,
#     format="%f",)
# 
LearningRate=0.001
st.title("Demo Deep Neural Network")
if file:
    df=pd.read_csv(file)
    st.write('Data',df.head())
    
    properties=st.selectbox("Select label",df.columns)

    
    
    if properties:
        X=df.drop(columns=[properties])
        
        Y=df[[properties]]
        
        X_train, X_test, Y_train, Y_test =train_test_split(X,Y,test_size=0.2)
        
        
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(len(X.columns), activation=activation_input,input_shape=(len(X.columns),)))
    
        for i in range(0,hiddenlayer):
            model.add(keras.layers.Dense(node[i], activation=acc[i]))
        
        model.add(keras.layers.Dense(1,activation=activation_output))
        dnn=model.summary()
       
        opt=keras.optimizers.Adam(learning_rate=LearningRate)
        model.compile(optimizer= opt, loss='mean_squared_error', metrics='accuracy')
        model.fit(X_train,Y_train,epochs=100,callbacks=[keras.callbacks.EarlyStopping(patience=3)],validation_split=0.3)
        st.success('train model successful')
        model.summary(print_fn=lambda x:st.text(x))
        # 
        st.title("Input")
        so=[]
        for i in range(0,len(X.columns)):
            temp=st.number_input(str(X.columns[i]),min_value=0.000,
            step=0.0001,
            value=0.0001,
            format="%f",)
            so.append(temp)
        

        test_data=np.array(so)
        

        
        a=model.predict(test_data.reshape(1,8),batch_size=1)
        click=st.button('Result')
        if click:
            st.success(a)
