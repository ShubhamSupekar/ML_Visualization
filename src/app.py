import streamlit as st

# Title of the app
st.title("Basic Streamlit App")

# Text input example
name = st.text_input("Enter your name:")

# Number input example
age = st.number_input("Enter your age:", min_value=0, max_value=120)

# Button to submit
if st.button("Submit"):
    if name:
        st.write(f"Hello, {name}! You are {age} years old.")
    else:
        st.write("Please enter your name.")
