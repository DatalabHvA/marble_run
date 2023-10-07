# Import Streamlit and other necessary libraries
import streamlit as st
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def encode_marble(param4):
	# Encode the categorical parameter for prediction
	param4_encoded = 1 if param4 == "A" else 0
	return param4_encoded

def get_predictions(model_lm, model_rf, param1,param2,param3, param4):

	param4_encoded = encode_marble(param4)
	prediction_lm = model_lm.predict([[param1, param2, param3, param4_encoded]])[0]
	prediction_rf = model_rf.predict([[param1, param2, param3, param4_encoded]])[0]
	return prediction_lm, prediction_rf

def update_models(model_lm, model_rf, data, param1, param2, param3, param4, actual_time):
	
	param4_encoded = encode_marble(param4)
	X_new = pd.DataFrame([[param1, param2, param3, param4_encoded]], columns=['recht','gebogen','pijpje', 'knikker_encoded'])
	y_new = pd.Series([actual_time])
	data = pd.concat([data,X_new.assign(tijd=y_new)], ignore_index=True)
    
	# Retrain the models
	X = data[['recht','gebogen','pijpje', 'knikker_encoded']]
	y = data['tijd']
	model_lm.fit(X, y)
	model_rf.fit(X, y)

	joblib.dump(model_lm, 'marble_run_model_lm.pkl')
	joblib.dump(model_rf, 'marble_run_model_rf.pkl')
	data[['recht','gebogen','pijpje', 'knikker_encoded', 'tijd']].to_csv('marble_run_data.csv')

	return model_lm, model_rf, data

# Load the pre-trained model
model_lm = joblib.load('marble_run_model_lm.pkl')
model_rf = joblib.load('marble_run_model_rf.pkl')
data = pd.read_csv('marble_run_data.csv')

# Streamlit app header
st.title("Marble Run Experiment Demo")

# User input for parameters
param1 = st.slider("# rechte bruggen", min_value=0, max_value=10, step=1)
param2 = st.slider("# aantal gebogen bruggen", min_value=0, max_value=10, step=1)
param3 = st.slider("# aantal kleine pijpjes", min_value=0, max_value=10, step=1)
param4 = st.selectbox("type knikker", ('A','B'))

# Prediction
prediction_lm, prediction_rf = get_predictions(model_lm, model_rf, param1, param2, param3, param4)
st.write(f"Predicted Time linear model: {prediction_lm:.2f} seconds")
st.write(f"Predicted Time random forest: {prediction_rf:.2f} seconds")

# User input for actual time
actual_time = st.number_input("Actual Time (seconds)")

# Update the model with new data
if st.button("Update Model"):
	model_lm, model_rf, data = update_models(model_lm, model_rf, data, param1, param2, param3, param4, actual_time)
	# Save the updated model
	st.success("Model updated successfully!")

# Display predictions vs. parameters
st.subheader("Predicted Time vs. Parameters")

data_pred = pd.DataFrame({'recht' : range(0, data['recht'].max()+1,1),'gebogen' : np.floor(data['gebogen'].mean()),'pijpje' : np.floor(data['pijpje'].mean()),'knikker_encoded' : data['knikker_encoded'].max()})
data_pred['pred_lm'] = model_lm.predict(data_pred[['recht','gebogen','pijpje', 'knikker_encoded']])
data_pred['pred_rf'] = model_rf.predict(data_pred[['recht','gebogen','pijpje', 'knikker_encoded']])

fig = plt.figure(figsize=(10, 4))
sns.lineplot(data = data_pred, x = 'recht', y = 'pred_lm')
sns.lineplot(data = data_pred, x = 'recht', y = 'pred_rf')
st.pyplot(fig)


#st.subheader("Predicted Time vs. Parameters")
#fig2 = show_error(data)
#st.figshow(fig2)