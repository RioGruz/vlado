import streamlit as st 
import pandas as pd
import pickle
import numpy as np

def main():
	st.title("XGBoost for LightShip prediction")
	# load saved model
	with open('model_pkl' , 'rb') as f:
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
