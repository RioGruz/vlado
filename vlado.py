import streamlit as st 
import pandas as pd
import pickle
import numpy as np
import json
from streamlit_lottie import st_lottie

def main():

	def load_lottiefile(filepath: str):
		with open (filepath, "r") as f:
			return json.load(f)

	lottie_coding = load_lottiefile("ship.json")
	st_lottie(lottie_coding, speed=1, reverse=False, loop=True, height=250, width=500)
	st.title("XGBoost for LightShip prediction")

	# load saved model
	with open('model_0_pkl' , 'rb') as f:
		lr = pickle.load(f)
	LOA = st.number_input("Enter LOA")
	B = st.number_input("Enter B")

	if st.button("Predict LightShip"):
		lista = LOA, B 
		x = np.array(lista).reshape((1,-1))
		a=lr.predict(x)
		st.write("LightShip is:",np.round(a,2))
	else:
		st.write("Please input LOA and B numbers and press Enter")

if __name__ == '__main__':
	main()