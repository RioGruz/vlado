# Uvoz librarya

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Postavljanje naslova
st.title("LightShip Model")

# Upload dataseta

st.header("XGBoost machine learning algoritam - trening i rezultati")

upload = st.file_uploader("Izaberi Excel file", type='xlsx')

if upload is not None:
	df=pd.read_excel(upload)
	st.write(df)
	st.success("Uspješan upload")
	X = df.loc[:,["LOA","B"]].values
	y = df.loc[:,"Lightship"].values

	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split (X,y, test_size = 0.15, random_state=0)

	import xgboost as xgb
	
	model = xgb.XGBRegressor()
	model.fit(X_train, y_train)

	y_pred = model.predict(X_test)
	np.set_printoptions(precision=2)
	print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

	from sklearn.metrics import accuracy_score

	from sklearn.metrics import r2_score
	from sklearn.metrics import mean_squared_error
	error = mean_squared_error(y_test, y_pred, squared=False)
	r2 = r2_score(y_test, y_pred)

	st.header("Algoritam XGBoost postigao je sljedeće rezultate: ")

	st.write("RMSE je ", np.round(error,2) )
	st.write("R^2 (koeficijent determinacije) je: ", np.round(r2,4))

else:
	st.markdown("Molim uplodajte Excel file")

# Upload dataseta za test

st.text("----"*100)

st.header("Testiranje novih podataka na postojećem (treniranom) XGBoost algoritmu")

upload2 = st.file_uploader("Izaberi excel file", type='xlsx')

if upload2 is not None:
	Z=pd.read_excel(upload2)
	st.write(Z)
	st.success("Uspješan upload")

	Zy = Z[["LOA", "B"]] # formiranje seta

	Zy = Zy.values # pretvaranje u array (XGBoost očekuje array)

	yhat = model.predict(Zy)  # prediktivni model na novom setu (Zy)

	Z.Lightship  # vrijednosti iz originalnog novog seta podataka 

	W = pd.DataFrame(yhat, columns=["predikcije"])  

	Z.Lightship.to_frame()

	Q = pd.concat([W,Z.Lightship], axis=1)  # spajanje predikcija i originalnih podataka u jedan dataframe

	Q["razlika"] = Q.predikcije - Q.Lightship # stvaranje novog stupca, razlika

	Q["RSE"] = np.sqrt(Q["razlika"]**2)  # formiranje stupca drugog korijena kvadrata razlike

	Q # prikaz

	RMSE = np.mean(Q.RSE) # računanje RMSE

	st.write("RMSE je ", np.round(RMSE,2) )

else:
	st.markdown("Molim uplodajte Excel file")
		
