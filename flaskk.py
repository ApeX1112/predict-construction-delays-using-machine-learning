import pandas as pd
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify, render_template
from ensemble import ensemble  # Ensure this matches your ensemble model's structure
from sklearn.ensemble import RandomForestRegressor ,GradientBoostingRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor
app = Flask(__name__)

# Initialize your model
ensemble1=RandomForestRegressor(n_estimators=1000,random_state=12)
ensemble2=GradientBoostingRegressor(n_estimators=100,learning_rate=0.1,max_depth=5,random_state=42)
ensemble3=GaussianNB()
ensemble4=DecisionTreeRegressor()


data=pd.read_csv("data.csv")
X=data.drop("Delay time",axis=1)
y=data["Delay time"]
X_train ,X_test,y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)
model = ensemble(ensemble1,ensemble2,ensemble3,ensemble4,X_train,y_train,X_test,y_test)



print(model.r2_score_models())
@app.route('/')
def home():
    return render_template('interface.html')  # Serve your main web interface

@app.route('/predict', methods=['POST'])

def predict():
    val1=request.form.get("input1")
    val2=request.form.get("input3")
    '''
    val3=int(request.form["input4"])
    val4=int(request.form["input5"])
    val5=float(request.form["sliderInput"])
    val6=float(request.form["sliderInput1"])
    val7=float(request.form["sliderInput2"])
    val8=float(request.form["sliderInput3"])
    val9=float(request.form["sliderInput4"])
    '''

    #data=[[val1,val2,val3,val4,val5,val6,val7,val8,val8,val9]]
    #result=model.make_prediction(data)[0]

    return jsonify(val1)

if __name__ == "__main__":
    app.run(debug=True)
