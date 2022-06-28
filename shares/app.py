from flask import Flask, render_template, redirect, url_for, request, jsonify
from flask_pymongo import PyMongo
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import json
import pickle

# Create an instance of Flask
app = Flask(__name__)

# Load model from local Directory
scaler = MinMaxScaler(feature_range=(0,1))
model_in_bhp = load_model('./static/bhp_model.sav')
model_in_cba = load_model('./static/cba_model.sav')
model_in_csl = load_model('./static/csl_model.sav')
model_in_nab = load_model('./static/nab_model.sav')
model_in_wbc = load_model('./static/wbc_model.sav')
filename = 'finalcba_model_rfr.sav'
model_in_cba_rfr = pickle.load(open(filename, 'rb'))

# Read in the CSV file
df_cba = pd.read_csv("./static/data/cba.csv")
df_cba = df_cba.dropna()
df_cba = df_cba.iloc[-4250:]
df_bhp = pd.read_csv("./static/data/bhp.csv")
df_bhp = df_bhp.dropna()
df_bhp = df_bhp.iloc[-4250:]
df_csl = pd.read_csv("./static/data/csl.csv")
df_csl = df_csl.dropna()
df_csl = df_csl.iloc[-4250:]
df_nab = pd.read_csv("./static/data/nab.csv")
df_nab = df_nab.dropna()
df_nab = df_nab.iloc[-4250:]
df_wbc = pd.read_csv("./static/data/wbc.csv")
df_wbc = df_wbc.dropna()
df_wbc = df_wbc.iloc[-4250:]
dates_df = pd.read_csv("./static/data/dates.csv")
dates_df['Count'] = dates_df['Count'].fillna(0).astype(int)
close_sixty_val_bhp = df_bhp.iloc[-60:,4].values
last_sixty_bhp = close_sixty_val_bhp.reshape(-1,1)
close_sixty_val_cba = df_cba.iloc[-60:,4].values
last_sixty_cba = close_sixty_val_cba.reshape(-1,1)
close_sixty_val_csl = df_csl.iloc[-60:,4].values
last_sixty_csl = close_sixty_val_csl.reshape(-1,1)
close_sixty_val_nab = df_nab.iloc[-60:,4].values
last_sixty_nab = close_sixty_val_nab.reshape(-1,1)
close_sixty_val_wbc = df_wbc.iloc[-60:,4].values
last_sixty_wbc = close_sixty_val_wbc.reshape(-1,1)


# Use PyMongo to establish Mongo connection to the database which is named last
mongo = PyMongo(app, uri="mongodb://localhost:27017/sharesDB")

# Route to render HOMEPAGE index.html template
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/cba_data")
def cba_data():
    # Return data
    cba_dict = df_cba.to_dict(orient='records')
    print(jsonify(cba_dict))
    return jsonify(dict=cba_dict)

@app.route("/csl_data")
def csl_data():
    # Return data
    csl_dict = df_csl.to_dict(orient='records')
    return jsonify(dict=csl_dict)

@app.route("/bhp_data")
def bhp_data():
    # Return data
    bhp_dict = df_bhp.to_dict(orient='records')
    return jsonify(dict=bhp_dict)

@app.route("/nab_data")
def nab_data():
    # Return data
    nab_dict = df_nab.to_dict(orient='records')
    return jsonify(dict=nab_dict)

@app.route("/wbc_data")
def wbc_data():
    # Return data
    wbc_dict = df_wbc.to_dict(orient='records')
    return jsonify(dict=wbc_dict)


@app.route("/cba.html", methods=('GET','POST'))
def predict_cba():
    request_type = request.method
    if request_type == 'GET':
        m_dict=[{}]
        m_rfr=[{}]
        path1 = '../static/data/images/cba_graph.png'
        path2 = '../static/data/images/cba_tree.png'
        return render_template('cba.html',href=path1,hreftwo=path2,dict=m_dict,dtree=m_rfr)
    else:
        input = request.form['text']
        if input == "":
         input = 20
        else:
         input = int(input)
        
        rba = request.form['rba']
        fed = request.form['fed']
        cpi = request.form['cpi']
        rba = float(rba)
        fed = float(fed)
        cpi = float(cpi)
        rfr =[]
        def randomforest(rba,fed,cpi):
            for i in range(0,input):
                rfr.append([rba,fed,cpi])
                df = pd.DataFrame (rfr, columns = ['RBA','FED',"CPI"])
                pred_rf=model_in_cba_rfr.predict(df)
                df['Prediction'] = pred_rf
                df.round(3)
                my_rforest = df.to_dict(orient='records')
                for dict_value in my_rforest:
                    for k, v in dict_value.items():
                        dict_value[k] = round(v, 2)
                rba+=0.01
                fed+=0.01
                cpi+=0.01  
            else:
                print("Could not predict rfr!")
                print(input)
            print(my_rforest)
            return my_rforest        
        my_rf = randomforest(rba,fed,cpi)
        print(my_rf)
        
        price_list=[]
        def predict_cba(dates_df,last_sixty_cba,model_in_cba,input):
            for i in range(0, input):

                #Takes df and converts to model's predict shape
                last_sixty_scaled = scaler.fit_transform(last_sixty_cba)
                new_X_tell = []
                new_X_tell.append(last_sixty_scaled)
                new_X_tell = np.array(new_X_tell)
                new_X_tell = np.reshape(new_X_tell, (new_X_tell.shape[0], new_X_tell.shape[1],1))
                
                model_in_pd_scale = model_in_cba.predict(new_X_tell)
                model_in_price = scaler.inverse_transform(model_in_pd_scale) # New price predicted

                last_sixty_less_one = np.delete(last_sixty_cba, 0, 0)
                last_sixty_cba = np.append(last_sixty_less_one, model_in_price,axis = 0) # Update last 60
                print(i)
                print("Day finished! Price: ")
                price_float = float(model_in_price)
                price = round(price_float, 2)
                price_list.append(price)

            else:
                print("Could not predict further!")
                print(input)

        print(price_list)
        predict_cba(dates_df,last_sixty_cba,model_in_cba,input)
        dates_df_iloc = dates_df.iloc[0:input]
        dates_df_iloc['Price'] = price_list
        my_dict = dates_df_iloc.to_dict(orient='records')
        print(my_dict)
        path = '../static/data/images/cba_predict_graph.png'
        path2 = '../static/data/images/cba_tree.png'
        return render_template('cba.html',href=path,dict=my_dict,hreftwo=path2,dtree=my_rf)

@app.route('/bhp.html', methods=('GET','POST'))
def predict_bhp():
    request_type = request.method
    if request_type == 'GET':
        m_dict=[{}]
        return render_template('bhp.html', href='../static/data/images/bhp_graph.png',dict=m_dict)
    else:
        input = request.form['text']
        if input == "":
         input = 20
        else:
         input = int(input)
        
        price_list=[]
        def predict_bhp(dates_df,last_sixty_bhp,model_in_bhp,input):
            for i in range(0, input):

                #Takes df and converts to model's predict shape
                last_sixty_scaled = scaler.fit_transform(last_sixty_bhp)
                new_X_tell = []
                new_X_tell.append(last_sixty_scaled)
                new_X_tell = np.array(new_X_tell)
                new_X_tell = np.reshape(new_X_tell, (new_X_tell.shape[0], new_X_tell.shape[1],1))
                
                model_in_pd_scale = model_in_bhp.predict(new_X_tell)
                model_in_price = scaler.inverse_transform(model_in_pd_scale) # New price predicted

                last_sixty_less_one = np.delete(last_sixty_bhp, 0, 0)
                last_sixty_bhp = np.append(last_sixty_less_one, model_in_price,axis = 0) # Update last 60
                print(i)
                print("Day finished! Price: ")
                price_float = float(model_in_price)
                price = round(price_float, 2)
                price_list.append(price)

            else:
                print("Could not predict further!")
                print(input)

        print(price_list)
        path = '../static/data/images/bhp_predict_graph.png'
        predict_bhp(dates_df,last_sixty_bhp,model_in_bhp,input)

        dates_df_iloc = dates_df.iloc[0:input]
        dates_df_iloc['Price'] = price_list
        my_dict = dates_df_iloc.to_dict(orient='records')
        print(my_dict)
        return render_template('bhp.html', href=path, dict=my_dict)


@app.route('/csl.html', methods=('GET','POST'))
def predict_csl():

    request_type = request.method
    if request_type == 'GET':
        m_dict=[{}]
        return render_template('csl.html', href='../static/data/images/csl_graph.png',dict=m_dict)
    else:
        input = request.form['text']
        if input == "":
         input = 20
        else:
         input = int(input)
        
        price_list=[]
        def predict_csl(dates_df,last_sixty_csl,model_in_csl,input):
            for i in range(0, input):

                #Takes df and converts to model's predict shape
                last_sixty_scaled = scaler.fit_transform(last_sixty_csl)
                new_X_tell = []
                new_X_tell.append(last_sixty_scaled)
                new_X_tell = np.array(new_X_tell)
                new_X_tell = np.reshape(new_X_tell, (new_X_tell.shape[0], new_X_tell.shape[1],1))
                
                model_in_pd_scale = model_in_csl.predict(new_X_tell)
                model_in_price = scaler.inverse_transform(model_in_pd_scale) # New price predicted

                last_sixty_less_one = np.delete(last_sixty_csl, 0, 0)
                last_sixty_csl = np.append(last_sixty_less_one, model_in_price,axis = 0) # Update last 60
                print(i)
                print("Day finished! Price: ")
                price_float = float(model_in_price)
                price = round(price_float, 2)
                price_list.append(price)

            else:
                print("Could not predict further!")
                print(input)

        print(price_list)
        path = '../static/data/images/csl_predict_graph.png'
        predict_csl(dates_df,last_sixty_csl,model_in_csl,input)

        dates_df_iloc = dates_df.iloc[0:input]
        dates_df_iloc['Price'] = price_list
        my_dict = dates_df_iloc.to_dict(orient='records')
        print(my_dict)     
        return render_template('csl.html', href=path, dict=my_dict)


@app.route('/nab.html', methods=('GET','POST'))
def predict_nab():

    request_type = request.method
    if request_type == 'GET':
        m_dict=[{}]
        return render_template('nab.html', href='../static/data/images/nab_graph.png',dict=m_dict)
    else:
        input = request.form['text']
        if input == "":
         input = 20
        else:
         input = int(input)
        
        price_list=[]
        def predict_nab(dates_df,last_sixty_nab,model_in_nab,input):
            for i in range(0, input):

                #Takes df and converts to model's predict shape
                last_sixty_scaled = scaler.fit_transform(last_sixty_nab)
                new_X_tell = []
                new_X_tell.append(last_sixty_scaled)
                new_X_tell = np.array(new_X_tell)
                new_X_tell = np.reshape(new_X_tell, (new_X_tell.shape[0], new_X_tell.shape[1],1))
                
                model_in_pd_scale = model_in_nab.predict(new_X_tell)
                model_in_price = scaler.inverse_transform(model_in_pd_scale) # New price predicted

                last_sixty_less_one = np.delete(last_sixty_nab, 0, 0)
                last_sixty_nab = np.append(last_sixty_less_one, model_in_price,axis = 0) # Update last 60
                print(i)
                print("Day finished! Price: ")
                price_float = float(model_in_price)
                price = round(price_float, 2)
                price_list.append(price)

            else:
                print("Could not predict further!")
                print(input)

        print(price_list)
        path = '../static/data/images/nab_predict_graph.png'
        predict_nab(dates_df,last_sixty_nab,model_in_nab,input)

        dates_df_iloc = dates_df.iloc[0:input]
        dates_df_iloc['Price'] = price_list
        my_dict = dates_df_iloc.to_dict(orient='records')
        print(my_dict)
        return render_template('nab.html', href=path, dict=my_dict)
    

@app.route('/wbc.html', methods=('GET','POST'))
def predict_wbc():
    request_type = request.method
    if request_type == 'GET':
        m_dict=[{}]
        return render_template('wbc.html', href='../static/data/images/wbc_graph.png',dict=m_dict)
    else:
        input = request.form['text']
        if input == "":
         input = 20
        else:
         input = int(input)
        
        price_list=[]
        def predict_wbc(dates_df,last_sixty_wbc,model_in_wbc,input):
            for i in range(0, input):

                #Takes df and converts to model's predict shape
                last_sixty_scaled = scaler.fit_transform(last_sixty_wbc)
                new_X_tell = []
                new_X_tell.append(last_sixty_scaled)
                new_X_tell = np.array(new_X_tell)
                new_X_tell = np.reshape(new_X_tell, (new_X_tell.shape[0], new_X_tell.shape[1],1))
                
                model_in_pd_scale = model_in_wbc.predict(new_X_tell)
                model_in_price = scaler.inverse_transform(model_in_pd_scale) # New price predicted

                last_sixty_less_one = np.delete(last_sixty_wbc, 0, 0)
                last_sixty_wbc = np.append(last_sixty_less_one, model_in_price,axis = 0) # Update last 60
                print(i)
                print("Day finished! Price: ")
                price_float = float(model_in_price)
                price = round(price_float, 2)
                price_list.append(price)

            else:
                print("Could not predict further!")
                print(input)

        print(price_list)
        path = '../static/data/images/wbc_predict_graph.png'
        predict_wbc(dates_df,last_sixty_wbc,model_in_wbc,input)

        dates_df_iloc = dates_df.iloc[0:input]
        dates_df_iloc['Price'] = price_list
        my_dict = dates_df_iloc.to_dict(orient='records')
        print(my_dict)
        return render_template('wbc.html', href=path, dict=my_dict)
   

# # Route that button will trigger the scrape function
# @app.route("/scrape")

# def scrape():
    # #same as above find one to initiate mongo database
    # s_data = mongo.db.companys
    # .find_one()

    # # Run the scrape function and save the results to a variable
    # var_data = scrape_shares.scrape_info()

    # # Update the Mongo database using update and upsert=True
    # mongo.db.collection.update_one({}, {"$set": var_data}, upsert=True)
   
    #Redirect back to home page
    return redirect("/", code=302)

if __name__ == "__main__":
    app.run(debug=True)

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', debug = False)
