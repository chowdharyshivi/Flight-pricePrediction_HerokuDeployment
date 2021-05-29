import numpy as np
import pandas as pd
import sklearn
from flask import Flask, request, render_template
import pickle

app= Flask(__name__)
model=pickle.load(open('flight_price_pred.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods = ["GET","POST"])
def predict():
    if request.method=="POST":
        dep_date=request.form["Date_of_Journey"]
        Journey_Day = int(pd.to_datetime(dep_date, format="%Y-%m-%dT%H:%M").day)
        Journey_Month = int(pd.to_datetime(dep_date, format ="%Y-%m-%dT%H:%M").month)
        Dep_Time_hour = int(pd.to_datetime(dep_date, format ="%Y-%m-%dT%H:%M").hour)
        Dep_Time_minute = int(pd.to_datetime(dep_date, format ="%Y-%m-%dT%H:%M").minute)

        date_arr = request.form["Arrival_Time"]
        Arrival_Time_hour = int(pd.to_datetime(date_arr, format ="%Y-%m-%dT%H:%M").hour)
        Arrival_Time_minute = int(pd.to_datetime(date_arr, format ="%Y-%m-%dT%H:%M").minute)
    
        Duration_hours = abs(Arrival_Time_hour - Dep_Time_hour)
        Duration_mins = abs(Arrival_Time_minute - Dep_Time_minute)

        Total_Stops = int(request.form["Total_Stops"])

        Source = request.form["Source"]
        if(Source=='Delhi'):
            s_Delhi=1
            s_Mumbai=0
            s_Chennai=0
            s_Kolkata=0
    
        elif(Source=='Mumbai'):
            s_Delhi=0
            s_Mumbai=1
            s_Chennai=0
            s_Kolkata=0
        elif(Source=='Chennai'):
            s_Delhi=0
            s_Mumbai=0
            s_Chennai=1
            s_Kolkata=0
        else:
            s_Delhi=0
            s_Mumbai=0
            s_Chennai=0
            s_Kolkata=1
       

        Destination = request.form["Destination"]
        if(Destination=='Cochin'):
            d_Cochin = 1
            d_Delhi = 0
            d_New_Delhi = 0    
            d_Hyderabad = 0
            d_Kolkata = 0
            
        elif(Destination=='Delhi'):
            d_Cochin = 0
            d_Delhi = 1
            d_New_Delhi = 0    
            d_Hyderabad = 0
            d_Kolkata = 0
           
        elif(Destination=='New Delhi'):
            d_Cochin = 0
            d_Delhi = 0
            d_New_Delhi = 1   
            d_Hyderabad = 0
            d_Kolkata = 0
            
        elif(Destination=='Hyderabad'):
            d_Cochin = 0
            d_Delhi = 0
            d_New_Delhi = 0    
            d_Hyderabad = 1
            d_Kolkata = 0
           
        else:
            d_Cochin = 0
            d_Delhi = 0
            d_New_Delhi = 0    
            d_Hyderabad = 0
            d_Kolkata = 1
        
        Airline=request.form['Airline']
        if(Airline=='Jet Airways'):
            Jet_Airways = 1
            IndiGo = 0
            Air_India = 0
            Air_Asia = 0
            Multiple_carriers = 0
            SpiceJet = 0
            Vistara = 0
            GoAir = 0
            Multiple_carriers_Premium_economy = 0
            Jet_Airways_Business = 0
            Vistara_Premium_economy = 0
            Trujet = 0 

        elif (Airline=='IndiGo'):
            Jet_Airways = 0
            IndiGo = 1
            Air_India = 0
            Air_Asia = 0
            Multiple_carriers = 0
            SpiceJet = 0
            Vistara = 0
            GoAir = 0
            Multiple_carriers_Premium_economy = 0
            Jet_Airways_Business = 0
            Vistara_Premium_economy = 0
            Trujet = 0 

        elif (Airline=='Air India'):
            Jet_Airways = 0
            IndiGo = 0
            Air_India = 1
            Air_Asia = 0
            Multiple_carriers = 0
            SpiceJet = 0
            Vistara = 0
            GoAir = 0
            Multiple_carriers_Premium_economy = 0
            Jet_Airways_Business = 0
            Vistara_Premium_economy = 0
            Trujet = 0 

        elif (Airline=='Multiple carriers'):
            Jet_Airways = 0
            IndiGo = 0
            Air_India = 0
            Air_Asia = 0
            Multiple_carriers = 1
            SpiceJet = 0
            Vistara = 0
            GoAir = 0
            Multiple_carriers_Premium_economy = 0
            Jet_Airways_Business = 0
            Vistara_Premium_economy = 0
            Trujet = 0 

        elif (Airline=='SpiceJet'):
            Jet_Airways = 0
            IndiGo = 0
            Air_India = 0
            Air_Asia = 0
            Multiple_carriers = 0
            SpiceJet = 1
            Vistara = 0
            GoAir = 0
            Multiple_carriers_Premium_economy = 0
            Jet_Airways_Business = 0
            Vistara_Premium_economy = 0
            Trujet = 0  

        elif (Airline=='Vistara'):
            Jet_Airways = 0
            IndiGo = 0
            Air_India = 0
            Air_Asia = 0
            Multiple_carriers = 0
            SpiceJet = 0
            Vistara = 1
            GoAir = 0
            Multiple_carriers_Premium_economy = 0
            Jet_Airways_Business = 0
            Vistara_Premium_economy = 0
            Trujet = 0 

        elif (Airline=='GoAir'):
            Jet_Airways = 0
            IndiGo = 0
            Air_India = 0
            Air_Asia = 0
            Multiple_carriers = 0
            SpiceJet = 0
            Vistara = 0
            GoAir = 1
            Multiple_carriers_Premium_economy = 0
            Jet_Airways_Business = 0
            Vistara_Premium_economy = 0
            Trujet = 0 

        elif (Airline=='Multiple carriers Premium economy'):
            Jet_Airways = 0
            IndiGo = 0
            Air_India = 0
            Air_Asia = 0
            Multiple_carriers = 0
            SpiceJet = 0
            Vistara = 0
            GoAir = 0
            Multiple_carriers_Premium_economy = 1
            Jet_Airways_Business = 0
            Vistara_Premium_economy = 0
            Trujet = 0 

        elif (Airline=='Jet Airways Business'):
            Jet_Airways = 0
            IndiGo = 0
            Air_India = 0
            Air_Asia = 0
            Multiple_carriers = 0
            SpiceJet = 0
            Vistara = 0
            GoAir = 0
            Multiple_carriers_Premium_economy = 0
            Jet_Airways_Business = 1
            Vistara_Premium_economy = 0
            Trujet = 0 

        elif (Airline=='Vistara Premium economy'):
            Jet_Airways = 0
            IndiGo = 0
            Air_India = 0
            Air_Asia = 0
            Multiple_carriers = 0
            SpiceJet = 0
            Vistara = 0
            GoAir = 0
            Multiple_carriers_Premium_economy = 0
            Jet_Airways_Business = 0
            Vistara_Premium_economy = 1
            Trujet = 0 

        elif (Airline=='Trujet'):
            Jet_Airways = 0
            IndiGo = 0
            Air_India = 0
            Air_Asia = 0
            Multiple_carriers = 0
            SpiceJet = 0
            Vistara = 0
            GoAir = 0
            Multiple_carriers_Premium_economy = 0
            Jet_Airways_Business = 0
            Vistara_Premium_economy = 0
            Trujet = 1 

        else:
            Jet_Airways = 0
            IndiGo = 0
            Air_India = 0
            Air_Asia = 1
            Multiple_carriers = 0
            SpiceJet = 0
            Vistara = 0
            GoAir = 0
            Multiple_carriers_Premium_economy = 0
            Jet_Airways_Business = 0
            Vistara_Premium_economy = 0
            Trujet = 0 

        prediction=model.predict([[
            Total_Stops,
            Journey_Day,
            Journey_Month,
            Dep_Time_hour,
            Dep_Time_minute,
            Arrival_Time_hour,
            Arrival_Time_minute,
            Duration_hours,
            Duration_mins,
            Air_India,
            GoAir,
            IndiGo,
            Jet_Airways,
            Jet_Airways_Business,
            Multiple_carriers,
            Multiple_carriers_Premium_economy,
            SpiceJet,
            Trujet,
            Vistara,
            Vistara_Premium_economy,
            s_Chennai,
            s_Delhi,
            s_Kolkata,
            s_Mumbai,
            d_Cochin,
            d_Delhi,
            d_Hyderabad,
            d_Kolkata,
            d_New_Delhi
        ]])

        output=round(prediction[0],2)
        return render_template('index.html',prediction_text="Flight price is Rs. {}".format(output))

    return render_template("index.html")
        



       # to_predict_list = request.from.to_dict()
       # to_predict_list = list(to_predict_list.values())
       # to_predict_list = list(map(int,to_predicct_list))
      #  result = ValuePredictor(to_predict_list)
      #  if int(result)==1:
      #      prediction = 

      #  return render_template("result.html",prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
    app.config['TEMPLATES_AUTO_RELOAD'] = True