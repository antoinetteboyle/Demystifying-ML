from tkinter.font import names
from flask import Flask, render_template, redirect, url_for, request
from flask_pymongo import PyMongo
import pandas as pd
import json
# import scrape_shares

# Create an instance of Flask
app = Flask(__name__)

# Use PyMongo to establish Mongo connection to the database which is named last
mongo = PyMongo(app, uri="mongodb://localhost:27017/sharesDB")

# Route to render index.html template using data from Mongo
@app.route("/")
def home():
    # Return template and data
    return render_template("index.html")

# Set route
@app.route('/mdata')
def index():
    # Store the entire team collection in a list
    share_list = list(mongo.db.companys.find())
    print(share_list) # prints in console

    # Return the template with the share_list passed in
    return render_template('index.html', shares=share_list)

@app.route('/cba.html')
def cba():
    return render_template('cba.html')

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

@app.route('/assets/data/cba.csv')
def scatter():
    data = './static/data/CBA.csv'
    df = pd.read_csv(data)
    chart_data = df.to_dict(orient='records')
    chart_data = json.dumps(chart_data, indent=2)
    data = {'chart_data': chart_data}
  
    return render_template('cba.html',data=data)

# @app.route('/name')
# def user():
#    who = request.args.get('#sel1')
#    short_list = list(mongo.db.companys['name'].find({ "name": who }))
#    print(short_list) # prints in console
#    return render_template("index.html", names=short_list)


# # Route that button will trigger the scrape function
# @app.route("/scrape")

# def scrape():
    # #same as above find one to initiate mongo database
    # s_data = mongo.db.collection.find_one()

    # # Run the scrape function and save the results to a variable
    # var_data = scrape_shares.scrape_info()

    # # Update the Mongo database using update and upsert=True
    # mongo.db.collection.update_one({}, {"$set": var_data}, upsert=True)
   
    # #Redirect back to home page
    # return redirect("/", code=302)

if __name__ == "__main__":
    app.run(debug=True)
