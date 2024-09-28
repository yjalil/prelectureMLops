import streamlit as st
import requests

st.title= " Taxifare Predict"
distance = st.number_input ('enter the distance :')
passenger_count = st.number_input ('enter Number of passengers :')
is_day = st.number_input ('Day-1 / Night-0 :')
button = st.button('predict')

if button :
	url = "api_url"
	Payload = { "distance" : distance,
				"passenger_count": passenger_count}
	Response= requests.post(url+f"distance={distance}&passenger_count={passeneger_count}")
