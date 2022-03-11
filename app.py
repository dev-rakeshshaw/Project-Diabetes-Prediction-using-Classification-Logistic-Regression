# importing the necessary dependencies
from flask import Flask, render_template, request,jsonify
from flask_cors import CORS,cross_origin
from sklearn.preprocessing import StandardScaler 
import pickle
import pandas as pd


app = Flask(__name__) # initializing a flask app



@app.route('/',methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")


@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
@cross_origin()
def index():

    if request.method == 'POST':   
        try:
  
            Pregnancies=float(request.form['Pregnancies'])
            Glucose = float(request.form['Glucose'])
            BloodPressure = float(request.form['BloodPressure'])
            SkinThickness = float(request.form['SkinThickness'])
            Insulin = float(request.form['Insulin'])
            BMI = float(request.form['BMI'])
            DiabetesPedigreeFunction= float(request.form['DiabetesPedigreeFunction'])
            Age = float(request.form['Age'])
   

           
            data = pd.read_csv("diabetes_filtered.csv")
            #y = data['Outcome']
            X =data.drop(columns=["Outcome"])
            X.columns


            scaler =StandardScaler()
            X_scaled = scaler.fit_transform(X)
            print(X_scaled)


            filename = 'finalized.pickle'
            loaded_model = pickle.load(open(filename, 'rb')) # loading the model file from the storage
            # predictions using the loaded model file
            prediction=loaded_model.predict(scaler.transform([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,BMI, DiabetesPedigreeFunction, Age]]))
            print('Prediction is', prediction)

            # showing the prediction results in a UI
            if(prediction==1):
                pred="Yes"
            else:
                pred="No"


            return render_template('results.html',prediction=pred)
        except Exception as e:

            print('The Exception message is: ',e)
            return 'something is wrong'
    # return render_template('results.html')
    else:
        return render_template('index.html')



if __name__ == "__main__":
    #app.run(host='127.0.0.1', port=8001, debug=True)
	app.run(debug=True) # running the app