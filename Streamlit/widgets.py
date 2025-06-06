import streamlit as st

st.title("Stream Text Input")

age=st.slider("Select your age",1,100,25)
st.write(f"Your age is {age}")

name=st.text_input("Enter your name:")
if name:
    st.write(f"Hello, {name}")


    
