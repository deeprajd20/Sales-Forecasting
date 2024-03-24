import pickle
import numpy as np
import streamlit as st
model = pickle.load(open('F:\\Model Deployment\\Sales Forecasting\\sales_model.pkl','rb'))

def sales_for(TV,Radio,Newspaper):
    feat_1 = Radio*TV
    inputs = np.array([TV,Radio,Newspaper,feat_1],ndmin = 2)
    result = model.predict(inputs)[0]
    return result

def main():
    # title of the application
    st.title('Sales prediction')

    TV = st.number_input('Enter your TV promotion value')
    Radio = st.number_input('Enter your Radio Promotion value')
    Newspaper = st.number_input('Enter your Newspaper promotion value')
    

    #output

    output = []

    if st.button('Predict Sales'):
        output = sales_for(TV,Radio,Newspaper)

    st.success(output)

if __name__ == '__main__':
    main()