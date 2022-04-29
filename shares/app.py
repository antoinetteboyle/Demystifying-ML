from flask import Flask, render_template, redirect, url_for, request, jsonify
from flask_pymongo import PyMongo
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

# Create an instance of Flask
app = Flask(__name__)

# Load model from local Directory
scaler = MinMaxScaler(feature_range=(0,1))
model_in = load_model('./static/cba_model.sav')
# Read in the CSV file
df_cba_sixty = pd.read_csv("./static/data/cba_sixty.csv")
close_sixty_val = df_cba_sixty[-60:].values
last_sixty = close_sixty_val.reshape(-1,1)


# Use PyMongo to establish Mongo connection to the database which is named last
mongo = PyMongo(app, uri="mongodb://localhost:27017/sharesDB")

@app.route("/cba.html", methods=('GET','POST'))
def predict_cba():
    request_type = request.method
    if request_type == 'GET':
        return render_template('cba.html', href='../static/data/images/cba_graph.png')
    else:
        input = request.form['text']
        if input == "":
         input = 1
        else:
         input = int(input)
        
        price_list=[]
        def predict_cba(last_sixty,model_in,input):
            for i in range(0, input):

                #Takes df and converts to model's predict shape
                last_sixty_scaled = scaler.fit_transform(last_sixty)
                new_X_tell = []
                new_X_tell.append(last_sixty_scaled)
                new_X_tell = np.array(new_X_tell)
                new_X_tell = np.reshape(new_X_tell, (new_X_tell.shape[0], new_X_tell.shape[1],1))
                
                model_in_pd_scale = model_in.predict(new_X_tell)
                model_in_price = scaler.inverse_transform(model_in_pd_scale) # New price predicted

                last_sixty_less_one = np.delete(last_sixty, 0, 0)
                last_sixty = np.append(last_sixty_less_one, model_in_price,axis = 0) # Update last 60
                print(i)
                print("Day finished! Price: ")
                price_float = float(model_in_price)
                price = round(price_float, 2)
                price_list.append(price)

            else:
                print("Could not predict further!")
                print(input)

        print(price_list)
        path = '../static/data/images/cba_graph.png'
        predict_cba(last_sixty,model_in,input)
        return render_template('cba.html', href=path, price_list=price_list)

# Route to render index.html template
@app.route("/")
def home():
    # Return template and data
    return render_template("index.html")

@app.route('/nab.html')
def nab():
    return render_template('nab.html')

@app.route('/wbc.html')
def wbc():
    return render_template('wbc.html')

@app.route('/bhp.html')
def bhp():
    return render_template('bhp.html')

@app.route('/csl.html')
def csl():
    return render_template('csl.html')

# # Route that button will trigger the scrape function
# @app.route("/scrape")

# def scrape():
    # #same as above find one to initiate mongo database
    # s_data = mongo.db.collection.find_one()

    # # Run the scrape function and save the results to a variable
    # var_data = scrape_shares.scrape_info()

    # # Update the Mongo database using update and upsert=True
    # mongo.db.collection.update_one({}, {"$set": var_data}, upsert=True)
   
    #Redirect back to home page
    return redirect("/", code=302)

if __name__ == "__main__":
    app.run(debug=True)
