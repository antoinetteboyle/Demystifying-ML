from flask import Flask, render_template, redirect, url_for, request, jsonify
from flask_pymongo import pymongo
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import json
import pickle
from datetime import date,datetime
todaydt = date.today()
today=todaydt.strftime('%d/%m/%Y')
today

# Create an instance of Flask
app = Flask(__name__)

# Load model from local Directory
scaler = MinMaxScaler(feature_range=(0,1))
model_in_bhp = load_model('./static/bhp_model.sav')
model_in_cba = load_model('./static/cba_model.sav')
model_in_csl = load_model('./static/csl_model.sav')
model_in_nab = load_model('./static/nab_model.sav')
model_in_wbc = load_model('./static/wbc_model.sav')
model_in_nic = load_model('./static/nic_model.sav')
model_in_min = load_model('./static/min_model.sav')
model_in_lyc = load_model('./static/lyc_model.sav')
model_in_nhc = load_model('./static/nhc_model.sav')
model_in_shl = load_model('./static/shl_model.sav')
model_in_mqg = load_model('./static/mqg_model.sav')
model_in_fmg = load_model('./static/fmg_model.sav')
model_in_wds = load_model('./static/wds_model.sav')
model_in_wes = load_model('./static/wes_model.sav')
model_in_wow = load_model('./static/wow_model.sav')
model_in_tcl = load_model('./static/tcl_model.sav')
model_in_gmg = load_model('./static/gmg_model.sav')
model_in_ncm = load_model('./static/ncm_model.sav')
model_in_sto = load_model('./static/sto_model.sav')
model_in_s32 = load_model('./static/s32_model.sav')
model_in_anz = load_model('./static/anz_model.sav')
model_in_all = load_model('./static/all_model.sav')
model_in_sol = load_model('./static/sol_model.sav')
model_in_pru = load_model('./static/pru_model.sav')

filename_cba = './static/cba_model_rfr.sav'
model_in_cba_rfr = pickle.load(open(filename_cba, 'rb'))
filename_nab = './static/nab_model_rfr.sav'
model_in_nab_rfr = pickle.load(open(filename_nab, 'rb'))
filename_bhp = './static/bhp_model_rfr.sav'
model_in_bhp_rfr = pickle.load(open(filename_bhp, 'rb'))
filename_csl = './static/csl_model_rfr.sav'
model_in_csl_rfr = pickle.load(open(filename_csl, 'rb'))
filename_wbc = './static/wbc_model_rfr.sav'
model_in_wbc_rfr = pickle.load(open(filename_wbc, 'rb'))
filename_nic = './static/nic_model_rfr.sav'
model_in_nic_rfr = pickle.load(open(filename_nic, 'rb'))
filename_min = './static/min_model_rfr.sav'
model_in_min_rfr = pickle.load(open(filename_min, 'rb'))
filename_lyc = './static/lyc_model_rfr.sav'
model_in_lyc_rfr = pickle.load(open(filename_lyc, 'rb'))
filename_nhc = './static/nhc_model_rfr.sav'
model_in_nhc_rfr = pickle.load(open(filename_nhc, 'rb'))
filename_shl = './static/shl_model_rfr.sav'
model_in_shl_rfr = pickle.load(open(filename_shl, 'rb'))
filename_mqg = './static/mqg_model_rfr.sav'
model_in_mqg_rfr = pickle.load(open(filename_mqg, 'rb'))
filename_fmg = './static/fmg_model_rfr.sav'
model_in_fmg_rfr = pickle.load(open(filename_fmg, 'rb'))
filename_wds = './static/wds_model_rfr.sav'
model_in_wds_rfr = pickle.load(open(filename_wds, 'rb'))
filename_wes = './static/wes_model_rfr.sav'
model_in_wes_rfr = pickle.load(open(filename_wes, 'rb'))
filename_wow = './static/wow_model_rfr.sav'
model_in_wow_rfr = pickle.load(open(filename_wow, 'rb'))
filename_tcl = './static/tcl_model_rfr.sav'
model_in_tcl_rfr = pickle.load(open(filename_tcl, 'rb'))
filename_gmg = './static/gmg_model_rfr.sav'
model_in_gmg_rfr = pickle.load(open(filename_gmg, 'rb'))
filename_ncm = './static/ncm_model_rfr.sav'
model_in_ncm_rfr = pickle.load(open(filename_ncm, 'rb'))
filename_sto = './static/sto_model_rfr.sav'
model_in_sto_rfr = pickle.load(open(filename_sto, 'rb'))
filename_s32 = './static/s32_model_rfr.sav'
model_in_s32_rfr = pickle.load(open(filename_s32, 'rb'))
filename_anz = './static/anz_model_rfr.sav'
model_in_anz_rfr = pickle.load(open(filename_anz, 'rb'))
filename_all = './static/all_model_rfr.sav'
model_in_all_rfr = pickle.load(open(filename_all, 'rb'))
filename_sol = './static/sol_model_rfr.sav'
model_in_sol_rfr = pickle.load(open(filename_sol, 'rb'))
filename_pru = './static/pru_model_rfr.sav'
model_in_pru_rfr = pickle.load(open(filename_pru, 'rb'))

# Use PyMongo to establish Mongo connection to the database and then the collection
# mongo = pymongo(app, uri="mongodb://localhost:27017/sharesDB/companys")
conn = 'mongodb://localhost:27017'
client = pymongo.MongoClient(conn)
shares_db = client.sharesDB  # connect to database
db = shares_db.companys   # connect to specific collection
db_bhp = client.sharesDB.bhp # connect to specific collection
db_cba = client.sharesDB.cba # connect to specific collection
db_nab = client.sharesDB.nab # connect to specific collection
db_wbc = client.sharesDB.wbc # connect to specific collection
db_csl = client.sharesDB.csl # connect to specific collection
db_nic = client.sharesDB.nic # connect to specific collection
db_min = client.sharesDB.min # connect to specific collection
db_lyc = client.sharesDB.lyc # connect to specific collection
db_nhc = client.sharesDB.nhc # connect to specific collection
db_shl = client.sharesDB.shl # connect to specific collection
db_mqg = client.sharesDB.mqg # connect to specific collection
db_fmg = client.sharesDB.fmg # connect to specific collection
db_wds = client.sharesDB.wds # connect to specific collection
db_wes = client.sharesDB.wes # connect to specific collection
db_wow = client.sharesDB.wow # connect to specific collection
db_tcl = client.sharesDB.tcl # connect to specific collection
db_gmg = client.sharesDB.gmg # connect to specific collection
db_ncm = client.sharesDB.ncm # connect to specific collection
db_sto = client.sharesDB.sto # connect to specific collection
db_s32 = client.sharesDB.s32 # connect to specific collection
db_anz = client.sharesDB.anz # connect to specific collection
db_all = client.sharesDB.all # connect to specific collection
db_sol = client.sharesDB.sol # connect to specific collection
db_pru = client.sharesDB.pru # connect to specific collection

db_dates = client.sharesDB.dates # connect to specific collection
db_scrape = client.sharesDB.scrape # connect to specific collection


df_mongo_bhp = pd.DataFrame(list(db_bhp.find()))
sixty_val_bhp = df_mongo_bhp.iloc[-60:,4].values
last_sixty_bhp = sixty_val_bhp.reshape(-1,1)

df_mongo_cba = pd.DataFrame(list(db_cba.find()))
sixty_val_cba = df_mongo_cba.iloc[-60:,4].values
last_sixty_cba = sixty_val_cba.reshape(-1,1)

df_mongo_nab = pd.DataFrame(list(db_nab.find()))
sixty_val_nab = df_mongo_nab.iloc[-60:,4].values
last_sixty_nab = sixty_val_nab.reshape(-1,1)

df_mongo_wbc = pd.DataFrame(list(db_wbc.find()))
sixty_val_wbc = df_mongo_wbc.iloc[-60:,4].values
last_sixty_wbc = sixty_val_wbc.reshape(-1,1)

df_mongo_csl = pd.DataFrame(list(db_csl.find()))
sixty_val_csl = df_mongo_csl.iloc[-60:,4].values
last_sixty_csl = sixty_val_csl.reshape(-1,1)

df_mongo_nic = pd.DataFrame(list(db_nic.find()))
sixty_val_nic = df_mongo_nic.iloc[-60:,4].values
last_sixty_nic = sixty_val_nic.reshape(-1,1)

df_mongo_min = pd.DataFrame(list(db_min.find()))
sixty_val_min = df_mongo_min.iloc[-60:,4].values
last_sixty_min = sixty_val_min.reshape(-1,1)

df_mongo_lyc = pd.DataFrame(list(db_lyc.find()))
sixty_val_lyc = df_mongo_lyc.iloc[-60:,4].values
last_sixty_lyc = sixty_val_lyc.reshape(-1,1)

df_mongo_nhc = pd.DataFrame(list(db_nhc.find()))
sixty_val_nhc = df_mongo_nhc.iloc[-60:,4].values
last_sixty_nhc = sixty_val_nhc.reshape(-1,1)

df_mongo_shl = pd.DataFrame(list(db_shl.find()))
sixty_val_shl = df_mongo_shl.iloc[-60:,4].values
last_sixty_shl = sixty_val_shl.reshape(-1,1)

df_mongo_mqg = pd.DataFrame(list(db_mqg.find()))
sixty_val_mqg = df_mongo_mqg.iloc[-60:,4].values
last_sixty_mqg = sixty_val_mqg.reshape(-1,1)

df_mongo_fmg = pd.DataFrame(list(db_fmg.find()))
sixty_val_fmg = df_mongo_fmg.iloc[-60:,4].values
last_sixty_fmg = sixty_val_fmg.reshape(-1,1)

df_mongo_wds = pd.DataFrame(list(db_wds.find()))
sixty_val_wds = df_mongo_wds.iloc[-60:,4].values
last_sixty_wds = sixty_val_wds.reshape(-1,1)

df_mongo_wes = pd.DataFrame(list(db_wes.find()))
sixty_val_wes = df_mongo_wes.iloc[-60:,4].values
last_sixty_wes = sixty_val_wes.reshape(-1,1)

df_mongo_wow = pd.DataFrame(list(db_wow.find()))
sixty_val_wow = df_mongo_wow.iloc[-60:,4].values
last_sixty_wow = sixty_val_wow.reshape(-1,1)

df_mongo_tcl = pd.DataFrame(list(db_tcl.find()))
sixty_val_tcl = df_mongo_tcl.iloc[-60:,4].values
last_sixty_tcl = sixty_val_tcl.reshape(-1,1)

df_mongo_gmg = pd.DataFrame(list(db_gmg.find()))
sixty_val_gmg = df_mongo_gmg.iloc[-60:,4].values
last_sixty_gmg = sixty_val_gmg.reshape(-1,1)

df_mongo_ncm = pd.DataFrame(list(db_ncm.find()))
sixty_val_ncm = df_mongo_ncm.iloc[-60:,4].values
last_sixty_ncm = sixty_val_ncm.reshape(-1,1)

df_mongo_sto = pd.DataFrame(list(db_sto.find()))
sixty_val_sto = df_mongo_sto.iloc[-60:,4].values
last_sixty_sto = sixty_val_sto.reshape(-1,1)

df_mongo_s32 = pd.DataFrame(list(db_s32.find()))
sixty_val_s32 = df_mongo_s32.iloc[-60:,4].values
last_sixty_s32 = sixty_val_s32.reshape(-1,1)

df_mongo_anz = pd.DataFrame(list(db_anz.find()))
sixty_val_anz = df_mongo_anz.iloc[-60:,4].values
last_sixty_anz = sixty_val_anz.reshape(-1,1)

df_mongo_all = pd.DataFrame(list(db_all.find()))
sixty_val_all = df_mongo_all.iloc[-60:,4].values
last_sixty_all = sixty_val_all.reshape(-1,1)

df_mongo_sol = pd.DataFrame(list(db_sol.find()))
sixty_val_sol = df_mongo_sol.iloc[-60:,4].values
last_sixty_sol = sixty_val_sol.reshape(-1,1)

df_mongo_pru = pd.DataFrame(list(db_pru.find()))
sixty_val_pru = df_mongo_pru.iloc[-60:,4].values
last_sixty_pru = sixty_val_pru.reshape(-1,1)

# dates_df = pd.read_csv("./static/data/dates.csv")

dates_df = pd.DataFrame(list(db_dates.find()))
dates_df = dates_df.iloc[0:180,1:]
# dates_df['Count'] = dates_df['Count'].fillna(0).astype(int)

# Read in the CSV file for the scatter plot
df_cba = pd.read_csv("./static/data/cba.csv")
df_cba = df_cba.dropna()
# df_cba = df_mongo_cba.dropna()
# df_cba = df_cba.iloc[-4250:]
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




def randomforest(model,rba,fed,cpi):
            rfr =[]
            for i in range(0,10):
                rfr.append([rba,fed,cpi])
                df = pd.DataFrame (rfr, columns = ['RBA','FED',"CPI"])
                pred_rf=model.predict(df)
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
                print(i)
            print(my_rforest)
            return my_rforest 

def predict(dates_df,last_sixty,model_in,input):
            price_list=[]
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
            dates_df_iloc = dates_df.iloc[0:input]
            dates_df_iloc['Price'] = price_list 
            my_dict = dates_df_iloc.to_dict(orient='records')
            return my_dict

# Route to render HOMEPAGE index.html template
@app.route("/")
def home():
    cba_lstm = db.find_one({'model': 'LSTM','name': 'CBA'})
    cba_rf = db.find_one({'model': 'RFR','name': 'CBA'})
    bhp_lstm = db.find_one({'model': 'LSTM','name': 'BHP'})
    bhp_rf = db.find_one({'model': 'RFR','name': 'BHP'})
    csl_lstm = db.find_one({'model': 'LSTM','name': 'CSL'})
    csl_rf = db.find_one({'model': 'RFR','name': 'CSL'})
    nab_lstm = db.find_one({'model': 'LSTM','name': 'NAB'})
    nab_rf = db.find_one({'model': 'RFR','name': 'NAB'})
    wbc_lstm = db.find_one({'model': 'LSTM','name': 'WBC'})
    wbc_rf = db.find_one({'model': 'RFR','name': 'WBC'})
    lyc_lstm = db.find_one({'model': 'LSTM','name': 'LYC'})
    lyc_rf = db.find_one({'model': 'RFR','name': 'LYC'})
    min_lstm = db.find_one({'model': 'LSTM','name': 'MIN'})
    min_rf = db.find_one({'model': 'RFR','name': 'MIN'})
    mqg_lstm = db.find_one({'model': 'LSTM','name': 'MQG'})
    mqg_rf = db.find_one({'model': 'RFR','name': 'MQG'})
    nhc_lstm = db.find_one({'model': 'LSTM','name': 'NHC'})
    nhc_rf = db.find_one({'model': 'RFR','name': 'NHC'})
    nic_lstm = db.find_one({'model': 'LSTM','name': 'NIC'})
    nic_rf = db.find_one({'model': 'RFR','name': 'NIC'})
    shl_lstm = db.find_one({'model': 'LSTM','name': 'SHL'})
    shl_rf = db.find_one({'model': 'RFR','name': 'SHL'})
    fmg_lstm = db.find_one({'model': 'LSTM','name': 'FMG'})
    fmg_rf = db.find_one({'model': 'RFR','name': 'FMG'})
    wds_lstm = db.find_one({'model': 'LSTM','name': 'WDS'})
    wds_rf = db.find_one({'model': 'RFR','name': 'WDS'})
    wes_lstm = db.find_one({'model': 'LSTM','name': 'WES'})
    wes_rf = db.find_one({'model': 'RFR','name': 'WES'})
    wow_lstm = db.find_one({'model': 'LSTM','name': 'WOW'})
    wow_rf = db.find_one({'model': 'RFR','name': 'WOW'})
    tcl_lstm = db.find_one({'model': 'LSTM','name': 'TCL'})
    tcl_rf = db.find_one({'model': 'RFR','name': 'TCL'})
    gmg_lstm = db.find_one({'model': 'LSTM','name': 'GMG'})
    gmg_rf = db.find_one({'model': 'RFR','name': 'GMG'})
    ncm_lstm = db.find_one({'model': 'LSTM','name': 'NCM'})
    ncm_rf = db.find_one({'model': 'RFR','name': 'NCM'})
    sto_lstm = db.find_one({'model': 'LSTM','name': 'STO'})
    sto_rf = db.find_one({'model': 'RFR','name': 'STO'})
    s32_lstm = db.find_one({'model': 'LSTM','name': 'S32'})
    s32_rf = db.find_one({'model': 'RFR','name': 'S32'})
    anz_lstm = db.find_one({'model': 'LSTM','name': 'ANZ'})
    anz_rf = db.find_one({'model': 'RFR','name': 'ANZ'})
    all_lstm = db.find_one({'model': 'LSTM','name': 'ALL'})
    all_rf = db.find_one({'model': 'RFR','name': 'ALL'})
    sol_lstm = db.find_one({'model': 'LSTM','name': 'SOL'})
    sol_rf = db.find_one({'model': 'RFR','name': 'SOL'})
    pru_lstm = db.find_one({'model': 'LSTM','name': 'PRU'})
    pru_rf = db.find_one({'model': 'RFR','name': 'PRU'})

    return render_template("index.html",cba_lstm=cba_lstm,cba_rf=cba_rf,
    bhp_lstm=bhp_lstm,bhp_rf=bhp_rf,
    csl_lstm=csl_lstm,csl_rf=csl_rf,
    nab_lstm=nab_lstm,nab_rf=nab_rf,
    wbc_lstm=wbc_lstm,wbc_rf=wbc_rf,
    lyc_lstm=lyc_lstm,lyc_rf=lyc_rf,
    min_lstm=min_lstm,min_rf=min_rf,
    mqg_lstm=mqg_lstm,mqg_rf=mqg_rf,
    nhc_lstm=nhc_lstm,nhc_rf=nhc_rf,
    nic_lstm=nic_lstm,nic_rf=nic_rf,
    shl_lstm=shl_lstm,shl_rf=shl_rf,
    fmg_lstm=fmg_lstm,fmg_rf=fmg_rf,
    wds_lstm=wds_lstm,wds_rf=wds_rf,
    wes_lstm=wes_lstm,wes_rf=wes_rf,
    wow_lstm=wow_lstm,wow_rf=wow_rf,
    tcl_lstm=tcl_lstm,tcl_rf=tcl_rf,
    gmg_lstm=gmg_lstm,gmg_rf=gmg_rf,
    ncm_lstm=ncm_lstm,ncm_rf=ncm_rf,
    sto_lstm=sto_lstm,sto_rf=sto_rf,
    s32_lstm=s32_lstm,s32_rf=s32_rf,
    anz_lstm=anz_lstm,anz_rf=anz_rf,
    all_lstm=all_lstm,all_rf=all_rf,
    sol_lstm=sol_lstm,sol_rf=sol_rf,
    pru_lstm=pru_lstm,pru_rf=pru_rf)

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
    cba_lstm = db.find_one({'model': 'LSTM','name': 'CBA'})
    cba_rf = db.find_one({'model': 'RFR','name': 'CBA'})
    request_type = request.method
    if request_type == 'GET':
        m_dict=[{}]
        m_rfr=[{}]
        path1 = '../static/data/images/cba_graph.png'
        path2 = '../static/data/images/cba_tree.png'
        path3 = '../static/data/images/cba.png'
        return render_template('cba.html',href=path1,hreftwo=path2,hrefthree=path3,dict=m_dict,dtree=m_rfr,cba=cba_lstm,rf=cba_rf)
    else:
        input = request.form['text']
        if input == "":
         input = 180
        else:
         input = int(input)
        
        rba = request.form['rba']
        fed = request.form['fed']
        cpi = request.form['cpi']
        rba = float(rba)
        fed = float(fed)
        cpi = float(cpi)
        model = model_in_cba_rfr 
        my_rf = randomforest(model,rba,fed,cpi)
        my_dict = predict(dates_df,last_sixty_cba,model_in_cba,input)
        path = '../static/data/images/cba_predict_graph.png'
        path2 = '../static/data/images/cba_tree.png'
        path3 = '../static/data/images/pred/cba_pred.png'
        return render_template('cba.html',href=path,dict=my_dict,hreftwo=path2,hrefthree=path3,dtree=my_rf,cba=cba_lstm,rf=cba_rf)

@app.route('/bhp.html', methods=('GET','POST'))
def predict_bhp():
    bhp_lstm = db.find_one({'model': 'LSTM','name': 'BHP','date':today})
    bhp_rf = db.find_one({'model': 'RFR','name': 'BHP'})
    request_type = request.method
    if request_type == 'GET':
        m_dict=[{}]
        m_rfr=[{}]
        path1 = '../static/data/images/bhp_graph.png'
        path2 = '../static/data/images/bhp_tree.png'
        path3 = '../static/data/images/bhp.png'
        return render_template('bhp.html', href=path1,hreftwo=path2,hrefthree=path3,dict=m_dict,dtree=m_rfr,bhp=bhp_lstm,rf=bhp_rf)
    else:
        input = request.form['text']
        if input == "":
         input = 180
        else:
         input = int(input) 

        rba = request.form['rba']
        fed = request.form['fed']
        cpi = request.form['cpi']
        rba = float(rba)
        fed = float(fed)
        cpi = float(cpi)
        model=model_in_bhp_rfr
        my_rf = randomforest(model,rba,fed,cpi)
        
        my_dict = predict(dates_df,last_sixty_bhp,model_in_bhp,input)
        path = '../static/data/images/bhp_predict_graph.png'
        path2 = '../static/data/images/bhp_tree.png'
        path3 = '../static/data/images/pred/bhp_pred.png'
        return render_template('bhp.html', href=path, dict=my_dict,hreftwo=path2,hrefthree=path3,dtree=my_rf,bhp=bhp_lstm,rf=bhp_rf)

@app.route('/csl.html', methods=('GET','POST'))
def predict_csl():
    csl_lstm = db.find_one({'model': 'LSTM','name': 'CSL'})
    csl_rf = db.find_one({'model': 'RFR','name': 'CSL'})
    request_type = request.method
    if request_type == 'GET':
        m_dict=[{}]
        m_rfr=[{}]
        path1 = '../static/data/images/csl_graph.png'
        path2 = '../static/data/images/csl_tree.png'
        path3 = '../static/data/images/csl.png'
        return render_template('csl.html', href=path1,hreftwo=path2,hrefthree=path3,dict=m_dict,dtree=m_rfr,csl=csl_lstm,rf=csl_rf)
    else:
        input = request.form['text']
        if input == "":
         input = 180
        else:
         input = int(input)
        
        rba = request.form['rba']
        fed = request.form['fed']
        cpi = request.form['cpi']
        rba = float(rba)
        fed = float(fed)
        cpi = float(cpi)
        model=model_in_csl_rfr   
        my_rf = randomforest(model,rba,fed,cpi)
        
        my_dict=predict(dates_df,last_sixty_csl,model_in_csl,input)
        path = '../static/data/images/csl_predict_graph.png'
        path2 = '../static/data/images/csl_tree.png'
        path3 = '../static/data/images/pred/csl_pred.png'
        return render_template('csl.html', href=path, dict=my_dict,hreftwo=path2,hrefthree=path3,dtree=my_rf,csl=csl_lstm,rf=csl_rf)


@app.route('/nab.html', methods=('GET','POST'))
def predict_nab():
    nab_lstm = db.find_one({'model': 'LSTM','name': 'NAB'})
    nab_rf = db.find_one({'model': 'RFR','name': 'NAB'})
    request_type = request.method
    if request_type == 'GET':
        m_dict=[{}]
        m_rfr=[{}]
        path1 = '../static/data/images/nab_graph.png'
        path2 = '../static/data/images/nab_tree.png'
        path3 = '../static/data/images/nab.png'
        return render_template('nab.html', href=path1,hreftwo=path2,hrefthree=path3,dict=m_dict,dtree=m_rfr,nab=nab_lstm,rf=nab_rf)
    else:
        input = request.form['text']
        if input == "":
         input = 180
        else:
         input = int(input)

        rba = request.form['rba']
        fed = request.form['fed']
        cpi = request.form['cpi']
        rba = float(rba)
        fed = float(fed)
        cpi = float(cpi)
        model=model_in_nab_rfr
        my_rf = randomforest(model,rba,fed,cpi)
        
        input=60
        my_dict=predict(dates_df,last_sixty_nab,model_in_nab,input)

        path = '../static/data/images/nab_predict_graph.png'
        path2 = '../static/data/images/nab_tree.png'
        path3 = '../static/data/images/pred/nab_pred.png'
        return render_template('nab.html', href=path, dict=my_dict,hreftwo=path2,hrefthree=path3,dtree=my_rf,nab=nab_lstm,rf=nab_rf)
    

@app.route('/wbc.html', methods=('GET','POST'))
def predict_wbc():
    wbc_lstm = db.find_one({'model': 'LSTM','name': 'WBC'})
    wbc_rf = db.find_one({'model': 'RFR','name': 'WBC'})
    request_type = request.method
    if request_type == 'GET':
        m_dict=[{}]
        m_rfr=[{}]
        path1 = '../static/data/images/wbc_graph.png'
        path2 = '../static/data/images/wbc_tree.png'
        path3 = '../static/data/images/wbc.png'
        return render_template('wbc.html', href=path1,hreftwo=path2,hrefthree=path3,dict=m_dict,dtree=m_rfr,wbc=wbc_lstm,rf=wbc_rf)
    else:
        input = request.form['text']
        if input == "":
         input = 180
        else:
         input = int(input)
        
        rba = request.form['rba']
        fed = request.form['fed']
        cpi = request.form['cpi']
        rba = float(rba)
        fed = float(fed)
        cpi = float(cpi)
        model=model_in_wbc_rfr
        my_rf = randomforest(model,rba,fed,cpi)

        my_dict=predict(dates_df,last_sixty_wbc,model_in_wbc,input)

        path = '../static/data/images/wbc_predict_graph.png'
        path2 = '../static/data/images/wbc_tree.png'
        path3 = '../static/data/images/pred/wbc_pred.png'
        return render_template('wbc.html', href=path, dict=my_dict,hreftwo=path2,hrefthree=path3,dtree=my_rf,wbc=wbc_lstm,rf=wbc_rf)
   
@app.route('/nic.html', methods=('GET','POST'))
def predict_nic():
    nic_lstm = db.find_one({'model': 'LSTM','name': 'NIC'})
    nic_rf = db.find_one({'model': 'RFR','name': 'NIC'})
    request_type = request.method
    if request_type == 'GET':
        m_dict=[{}]
        m_rfr=[{}]
        path1 = '../static/data/images/nic_graph.png'
        path2 = '../static/data/images/nic_tree.png'
        path3 = '../static/data/images/nic.png'
        return render_template('nic.html', href=path1,hreftwo=path2,hrefthree=path3,dict=m_dict,dtree=m_rfr,nic=nic_lstm,rf=nic_rf)
    else:
        input = request.form['text']
        if input == "":
         input = 180
        else:
         input = int(input)
        
        rba = request.form['rba']
        fed = request.form['fed']
        cpi = request.form['cpi']
        rba = float(rba)
        fed = float(fed)
        cpi = float(cpi)
        model=model_in_nic_rfr
        my_rf = randomforest(model,rba,fed,cpi)

        my_dict=predict(dates_df,last_sixty_nic,model_in_nic,input)

        path = '../static/data/images/nic_predict_graph.png'
        path2 = '../static/data/images/nic_tree.png'
        path3 = '../static/data/images/pred/nic_pred.png'
        return render_template('nic.html', href=path, dict=my_dict,hreftwo=path2,hrefthree=path3,dtree=my_rf,nic=nic_lstm,rf=nic_rf)

@app.route('/min.html', methods=('GET','POST'))
def predict_min():
    min_lstm = db.find_one({'model': 'LSTM','name': 'MIN'})
    min_rf = db.find_one({'model': 'RFR','name': 'MIN'})
    request_type = request.method
    if request_type == 'GET':
        m_dict=[{}]
        m_rfr=[{}]
        path1 = '../static/data/images/min_graph.png'
        path2 = '../static/data/images/min_tree.png'
        path3 = '../static/data/images/min.png'
        return render_template('min.html', href=path1,hreftwo=path2,hrefthree=path3,dict=m_dict,dtree=m_rfr,min=min_lstm,rf=min_rf)
    else:
        input = request.form['text']
        if input == "":
         input = 180
        else:
         input = int(input)
        
        rba = request.form['rba']
        fed = request.form['fed']
        cpi = request.form['cpi']
        rba = float(rba)
        fed = float(fed)
        cpi = float(cpi)
        model=model_in_min_rfr
        my_rf = randomforest(model,rba,fed,cpi)

        my_dict=predict(dates_df,last_sixty_min,model_in_min,input)

        path = '../static/data/images/min_predict_graph.png'
        path2 = '../static/data/images/min_tree.png'
        path3 = '../static/data/images/pred/min_pred.png'
        return render_template('min.html', href=path, dict=my_dict,hreftwo=path2,hrefthree=path3,dtree=my_rf,min=min_lstm,rf=min_rf)
     

@app.route('/lyc.html', methods=('GET','POST'))
def predict_lyc():
    lyc_lstm = db.find_one({'model': 'LSTM','name': 'LYC'})
    lyc_rf = db.find_one({'model': 'RFR','name': 'LYC'})
    request_type = request.method
    if request_type == 'GET':
        m_dict=[{}]
        m_rfr=[{}]
        path1 = '../static/data/images/lyc_graph.png'
        path2 = '../static/data/images/lyc_tree.png'
        path3 = '../static/data/images/lyc.png'
        return render_template('lyc.html', href=path1,hreftwo=path2,hrefthree=path3,dict=m_dict,dtree=m_rfr,lyc=lyc_lstm,rf=lyc_rf)
    else:
        input = request.form['text']
        if input == "":
         input = 180
        else:
         input = int(input)
        
        rba = request.form['rba']
        fed = request.form['fed']
        cpi = request.form['cpi']
        rba = float(rba)
        fed = float(fed)
        cpi = float(cpi)
        model=model_in_lyc_rfr
        my_rf = randomforest(model,rba,fed,cpi)

        my_dict=predict(dates_df,last_sixty_lyc,model_in_lyc,input)

        path = '../static/data/images/lyc_predict_graph.png'
        path2 = '../static/data/images/lyc_tree.png'
        path3 = '../static/data/images/pred/lyc_pred.png'
        return render_template('lyc.html', href=path, dict=my_dict,hreftwo=path2,hrefthree=path3,dtree=my_rf,lyc=lyc_lstm,rf=lyc_rf)


@app.route('/nhc.html', methods=('GET','POST'))
def predict_nhc():
    nhc_lstm = db.find_one({'model': 'LSTM','name': 'NHC'})
    nhc_rf = db.find_one({'model': 'RFR','name': 'NHC'})
    request_type = request.method
    if request_type == 'GET':
        m_dict=[{}]
        m_rfr=[{}]
        path1 = '../static/data/images/nhc_graph.png'
        path2 = '../static/data/images/nhc_tree.png'
        path3 = '../static/data/images/nhc.png'
        return render_template('nhc.html', href=path1,hreftwo=path2,hrefthree=path3,dict=m_dict,dtree=m_rfr,nhc=nhc_lstm,rf=nhc_rf)
    else:
        input = request.form['text']
        if input == "":
         input = 180
        else:
         input = int(input)
        
        rba = request.form['rba']
        fed = request.form['fed']
        cpi = request.form['cpi']
        rba = float(rba)
        fed = float(fed)
        cpi = float(cpi)
        model=model_in_nhc_rfr
        my_rf = randomforest(model,rba,fed,cpi)

        my_dict=predict(dates_df,last_sixty_nhc,model_in_nhc,input)

        path = '../static/data/images/nhc_predict_graph.png'
        path2 = '../static/data/images/nhc_tree.png'
        path3 = '../static/data/images/pred/nhc_pred.png'
        return render_template('nhc.html', href=path, dict=my_dict,hreftwo=path2,hrefthree=path3,dtree=my_rf,nhc=nhc_lstm,rf=nhc_rf)
     
@app.route('/shl.html', methods=('GET','POST'))
def predict_shl():
    shl_lstm = db.find_one({'model': 'LSTM','name': 'SHL'})
    shl_rf = db.find_one({'model': 'RFR','name': 'SHL'})
    request_type = request.method
    if request_type == 'GET':
        m_dict=[{}]
        m_rfr=[{}]
        path1 = '../static/data/images/shl_graph.png'
        path2 = '../static/data/images/shl_tree.png'
        path3 = '../static/data/images/shl.png'
        return render_template('shl.html', href=path1,hreftwo=path2,hrefthree=path3,dict=m_dict,dtree=m_rfr,shl=shl_lstm,rf=shl_rf)
    else:
        input = request.form['text']
        if input == "":
         input = 180
        else:
         input = int(input)
        
        rba = request.form['rba']
        fed = request.form['fed']
        cpi = request.form['cpi']
        rba = float(rba)
        fed = float(fed)
        cpi = float(cpi)
        model=model_in_shl_rfr
        my_rf = randomforest(model,rba,fed,cpi)

        my_dict=predict(dates_df,last_sixty_shl,model_in_shl,input)

        path = '../static/data/images/shl_predict_graph.png'
        path2 = '../static/data/images/shl_tree.png'
        path3 = '../static/data/images/pred/shl_pred.png'
        return render_template('shl.html', href=path, dict=my_dict,hreftwo=path2,hrefthree=path3,dtree=my_rf,shl=shl_lstm,rf=shl_rf)
     
@app.route('/mqg.html', methods=('GET','POST'))
def predict_mqg():
    mqg_lstm = db.find_one({'model': 'LSTM','name': 'MQG'})
    mqg_rf = db.find_one({'model': 'RFR','name': 'MQG'})
    request_type = request.method
    if request_type == 'GET':
        m_dict=[{}]
        m_rfr=[{}]
        path1 = '../static/data/images/mqg_graph.png'
        path2 = '../static/data/images/mqg_tree.png'
        path3 = '../static/data/images/mqg.png'
        return render_template('mqg.html', href=path1,hreftwo=path2,hrefthree=path3,dict=m_dict,dtree=m_rfr,mqg=mqg_lstm,rf=mqg_rf)
    else:
        input = request.form['text']
        if input == "":
         input = 180
        else:
         input = int(input)
        
        rba = request.form['rba']
        fed = request.form['fed']
        cpi = request.form['cpi']
        rba = float(rba)
        fed = float(fed)
        cpi = float(cpi)
        model=model_in_mqg_rfr
        my_rf = randomforest(model,rba,fed,cpi)

        my_dict=predict(dates_df,last_sixty_mqg,model_in_mqg,input)

        path = '../static/data/images/mqg_predict_graph.png'
        path2 = '../static/data/images/mqg_tree.png'
        path3 = '../static/data/images/pred/mqg_pred.png'
        return render_template('mqg.html', href=path, dict=my_dict,hreftwo=path2,hrefthree=path3,dtree=my_rf,mqg=mqg_lstm,rf=mqg_rf)

@app.route('/fmg.html', methods=('GET','POST'))
def predict_fmg():
    fmg_lstm = db.find_one({'model': 'LSTM','name': 'FMG'})
    fmg_rf = db.find_one({'model': 'RFR','name': 'FMG'})
    request_type = request.method
    if request_type == 'GET':
        m_dict=[{}]
        m_rfr=[{}]
        path1 = '../static/data/images/fmg_graph.png'
        path2 = '../static/data/images/fmg_tree.png'
        path3 = '../static/data/images/fmg.png'
        return render_template('fmg.html', href=path1,hreftwo=path2,hrefthree=path3,dict=m_dict,dtree=m_rfr,fmg=fmg_lstm,rf=fmg_rf)
    else:
        input = request.form['text']
        if input == "":
         input = 180
        else:
         input = int(input)
        
        rba = request.form['rba']
        fed = request.form['fed']
        cpi = request.form['cpi']
        rba = float(rba)
        fed = float(fed)
        cpi = float(cpi)
        model=model_in_fmg_rfr
        my_rf = randomforest(model,rba,fed,cpi)

        my_dict=predict(dates_df,last_sixty_fmg,model_in_fmg,input)

        path = '../static/data/images/fmg_predict_graph.png'
        path2 = '../static/data/images/fmg_tree.png'
        path3 = '../static/data/images/pred/fmg_pred.png'
        return render_template('fmg.html', href=path, dict=my_dict,hreftwo=path2,hrefthree=path3,dtree=my_rf,fmg=fmg_lstm,rf=fmg_rf)
     
@app.route('/wds.html', methods=('GET','POST'))
def predict_wds():
    wds_lstm = db.find_one({'model': 'LSTM','name': 'WDS'})
    wds_rf = db.find_one({'model': 'RFR','name': 'WDS'})
    request_type = request.method
    if request_type == 'GET':
        m_dict=[{}]
        m_rfr=[{}]
        path1 = '../static/data/images/wds_graph.png'
        path2 = '../static/data/images/wds_tree.png'
        path3 = '../static/data/images/wds.png'
        return render_template('wds.html', href=path1,hreftwo=path2,hrefthree=path3,dict=m_dict,dtree=m_rfr,wds=wds_lstm,rf=wds_rf)
    else:
        input = request.form['text']
        if input == "":
         input = 180
        else:
         input = int(input)
        
        rba = request.form['rba']
        fed = request.form['fed']
        cpi = request.form['cpi']
        rba = float(rba)
        fed = float(fed)
        cpi = float(cpi)
        model=model_in_wds_rfr
        my_rf = randomforest(model,rba,fed,cpi)

        my_dict=predict(dates_df,last_sixty_wds,model_in_wds,input)

        path = '../static/data/images/wds_predict_graph.png'
        path2 = '../static/data/images/wds_tree.png'
        path3 = '../static/data/images/pred/wds_pred.png'
        return render_template('wds.html', href=path, dict=my_dict,hreftwo=path2,hrefthree=path3,dtree=my_rf,wds=wds_lstm,rf=wds_rf)
     
@app.route('/wes.html', methods=('GET','POST'))
def predict_wes():
    wes_lstm = db.find_one({'model': 'LSTM','name': 'WES'})
    wes_rf = db.find_one({'model': 'RFR','name': 'WES'})
    request_type = request.method
    if request_type == 'GET':
        m_dict=[{}]
        m_rfr=[{}]
        path1 = '../static/data/images/wes_graph.png'
        path2 = '../static/data/images/wes_tree.png'
        path3 = '../static/data/images/wes.png'
        return render_template('wes.html', href=path1,hreftwo=path2,hrefthree=path3,dict=m_dict,dtree=m_rfr,wes=wes_lstm,rf=wes_rf)
    else:
        input = request.form['text']
        if input == "":
         input = 180
        else:
         input = int(input)
        
        rba = request.form['rba']
        fed = request.form['fed']
        cpi = request.form['cpi']
        rba = float(rba)
        fed = float(fed)
        cpi = float(cpi)
        model=model_in_wes_rfr
        my_rf = randomforest(model,rba,fed,cpi)

        my_dict=predict(dates_df,last_sixty_wes,model_in_wes,input)

        path = '../static/data/images/wes_predict_graph.png'
        path2 = '../static/data/images/wes_tree.png'
        path3 = '../static/data/images/pred/wes_pred.png'
        return render_template('wes.html', href=path, dict=my_dict,hreftwo=path2,hrefthree=path3,dtree=my_rf,wes=wes_lstm,rf=wes_rf)
     
@app.route('/wow.html', methods=('GET','POST'))
def predict_wow():
    wow_lstm = db.find_one({'model': 'LSTM','name': 'WOW'})
    wow_rf = db.find_one({'model': 'RFR','name': 'WOW'})
    request_type = request.method
    if request_type == 'GET':
        m_dict=[{}]
        m_rfr=[{}]
        path1 = '../static/data/images/wow_graph.png'
        path2 = '../static/data/images/wow_tree.png'
        path3 = '../static/data/images/wow.png'
        return render_template('wow.html', href=path1,hreftwo=path2,hrefthree=path3,dict=m_dict,dtree=m_rfr,wow=wow_lstm,rf=wow_rf)
    else:
        input = request.form['text']
        if input == "":
         input = 180
        else:
         input = int(input)
        
        rba = request.form['rba']
        fed = request.form['fed']
        cpi = request.form['cpi']
        rba = float(rba)
        fed = float(fed)
        cpi = float(cpi)
        model=model_in_wow_rfr
        my_rf = randomforest(model,rba,fed,cpi)

        my_dict=predict(dates_df,last_sixty_wow,model_in_wow,input)

        path = '../static/data/images/wow_predict_graph.png'
        path2 = '../static/data/images/wow_tree.png'
        path3 = '../static/data/images/pred/wow_pred.png'
        return render_template('wow.html', href=path, dict=my_dict,hreftwo=path2,hrefthree=path3,dtree=my_rf,wow=wow_lstm,rf=wow_rf)

@app.route('/tcl.html', methods=('GET','POST'))
def predict_tcl():
    tcl_lstm = db.find_one({'model': 'LSTM','name': 'TCL'})
    tcl_rf = db.find_one({'model': 'RFR','name': 'TCL'})
    request_type = request.method
    if request_type == 'GET':
        m_dict=[{}]
        m_rfr=[{}]
        path1 = '../static/data/images/tcl_graph.png'
        path2 = '../static/data/images/tcl_tree.png'
        path3 = '../static/data/images/tcl.png'
        return render_template('tcl.html', href=path1,hreftwo=path2,hrefthree=path3,dict=m_dict,dtree=m_rfr,tcl=tcl_lstm,rf=tcl_rf)
    else:
        input = request.form['text']
        if input == "":
         input = 180
        else:
         input = int(input)
        
        rba = request.form['rba']
        fed = request.form['fed']
        cpi = request.form['cpi']
        rba = float(rba)
        fed = float(fed)
        cpi = float(cpi)
        model=model_in_tcl_rfr
        my_rf = randomforest(model,rba,fed,cpi)

        my_dict=predict(dates_df,last_sixty_tcl,model_in_tcl,input)

        path = '../static/data/images/tcl_predict_graph.png'
        path2 = '../static/data/images/tcl_tree.png'
        path3 = '../static/data/images/pred/tcl_pred.png'
        return render_template('tcl.html', href=path, dict=my_dict,hreftwo=path2,hrefthree=path3,dtree=my_rf,tcl=tcl_lstm,rf=tcl_rf)
         
@app.route('/gmg.html', methods=('GET','POST'))
def predict_gmg():
    gmg_lstm = db.find_one({'model': 'LSTM','name': 'GMG'})
    gmg_rf = db.find_one({'model': 'RFR','name': 'GMG'})
    request_type = request.method
    if request_type == 'GET':
        m_dict=[{}]
        m_rfr=[{}]
        path1 = '../static/data/images/gmg_graph.png'
        path2 = '../static/data/images/gmg_tree.png'
        path3 = '../static/data/images/gmg.png'
        return render_template('gmg.html', href=path1,hreftwo=path2,hrefthree=path3,dict=m_dict,dtree=m_rfr,gmg=gmg_lstm,rf=gmg_rf)
    else:
        input = request.form['text']
        if input == "":
         input = 180
        else:
         input = int(input)
        
        rba = request.form['rba']
        fed = request.form['fed']
        cpi = request.form['cpi']
        rba = float(rba)
        fed = float(fed)
        cpi = float(cpi)
        model=model_in_gmg_rfr
        my_rf = randomforest(model,rba,fed,cpi)

        my_dict=predict(dates_df,last_sixty_gmg,model_in_gmg,input)

        path = '../static/data/images/gmg_predict_graph.png'
        path2 = '../static/data/images/gmg_tree.png'
        path3 = '../static/data/images/pred/gmg_pred.png'
        return render_template('gmg.html', href=path, dict=my_dict,hreftwo=path2,hrefthree=path3,dtree=my_rf,gmg=gmg_lstm,rf=gmg_rf)

@app.route('/ncm.html', methods=('GET','POST'))
def predict_ncm():
    ncm_lstm = db.find_one({'model': 'LSTM','name': 'NCM'})
    ncm_rf = db.find_one({'model': 'RFR','name': 'NCM'})
    request_type = request.method
    if request_type == 'GET':
        m_dict=[{}]
        m_rfr=[{}]
        path1 = '../static/data/images/ncm_graph.png'
        path2 = '../static/data/images/ncm_tree.png'
        path3 = '../static/data/images/ncm.png'
        return render_template('ncm.html', href=path1,hreftwo=path2,hrefthree=path3,dict=m_dict,dtree=m_rfr,ncm=ncm_lstm,rf=ncm_rf)
    else:
        input = request.form['text']
        if input == "":
         input = 180
        else:
         input = int(input)
        
        rba = request.form['rba']
        fed = request.form['fed']
        cpi = request.form['cpi']
        rba = float(rba)
        fed = float(fed)
        cpi = float(cpi)
        model=model_in_ncm_rfr
        my_rf = randomforest(model,rba,fed,cpi)

        my_dict=predict(dates_df,last_sixty_ncm,model_in_ncm,input)

        path = '../static/data/images/ncm_predict_graph.png'
        path2 = '../static/data/images/ncm_tree.png'
        path3 = '../static/data/images/pred/ncm_pred.png'
        return render_template('ncm.html', href=path, dict=my_dict,hreftwo=path2,hrefthree=path3,dtree=my_rf,ncm=ncm_lstm,rf=ncm_rf)
        
@app.route('/sto.html', methods=('GET','POST'))
def predict_sto():
    sto_lstm = db.find_one({'model': 'LSTM','name': 'STO'})
    sto_rf = db.find_one({'model': 'RFR','name': 'STO'})
    request_type = request.method
    if request_type == 'GET':
        m_dict=[{}]
        m_rfr=[{}]
        path1 = '../static/data/images/sto_graph.png'
        path2 = '../static/data/images/sto_tree.png'
        path3 = '../static/data/images/sto.png'
        return render_template('sto.html', href=path1,hreftwo=path2,hrefthree=path3,dict=m_dict,dtree=m_rfr,sto=sto_lstm,rf=sto_rf)
    else:
        input = request.form['text']
        if input == "":
         input = 180
        else:
         input = int(input)
        
        rba = request.form['rba']
        fed = request.form['fed']
        cpi = request.form['cpi']
        rba = float(rba)
        fed = float(fed)
        cpi = float(cpi)
        model=model_in_sto_rfr
        my_rf = randomforest(model,rba,fed,cpi)

        my_dict=predict(dates_df,last_sixty_sto,model_in_sto,input)

        path = '../static/data/images/sto_predict_graph.png'
        path2 = '../static/data/images/sto_tree.png'
        path3 = '../static/data/images/pred/sto_pred.png'
        return render_template('sto.html', href=path, dict=my_dict,hreftwo=path2,hrefthree=path3,dtree=my_rf,sto=sto_lstm,rf=sto_rf)
     
@app.route('/s32.html', methods=('GET','POST'))
def predict_s32():
    s32_lstm = db.find_one({'model': 'LSTM','name': 'S32'})
    s32_rf = db.find_one({'model': 'RFR','name': 'S32'})
    request_type = request.method
    if request_type == 'GET':
        m_dict=[{}]
        m_rfr=[{}]
        path1 = '../static/data/images/s32_graph.png'
        path2 = '../static/data/images/s32_tree.png'
        path3 = '../static/data/images/s32.png'
        return render_template('s32.html', href=path1,hreftwo=path2,hrefthree=path3,dict=m_dict,dtree=m_rfr,s32=s32_lstm,rf=s32_rf)
    else:
        input = request.form['text']
        if input == "":
         input = 180
        else:
         input = int(input)
        
        rba = request.form['rba']
        fed = request.form['fed']
        cpi = request.form['cpi']
        rba = float(rba)
        fed = float(fed)
        cpi = float(cpi)
        model=model_in_s32_rfr
        my_rf = randomforest(model,rba,fed,cpi)

        my_dict=predict(dates_df,last_sixty_s32,model_in_s32,input)

        path = '../static/data/images/s32_predict_graph.png'
        path2 = '../static/data/images/s32_tree.png'
        path3 = '../static/data/images/pred/s32_pred.png'
        return render_template('s32.html', href=path, dict=my_dict,hreftwo=path2,hrefthree=path3,dtree=my_rf,s32=s32_lstm,rf=s32_rf)
     
@app.route('/anz.html', methods=('GET','POST'))
def predict_anz():
    anz_lstm = db.find_one({'model': 'LSTM','name': 'ANZ'})
    anz_rf = db.find_one({'model': 'RFR','name': 'ANZ'})
    request_type = request.method
    if request_type == 'GET':
        m_dict=[{}]
        m_rfr=[{}]
        path1 = '../static/data/images/anz_graph.png'
        path2 = '../static/data/images/anz_tree.png'
        path3 = '../static/data/images/anz.png'
        return render_template('anz.html', href=path1,hreftwo=path2,hrefthree=path3,dict=m_dict,dtree=m_rfr,anz=anz_lstm,rf=anz_rf)
    else:
        input = request.form['text']
        if input == "":
         input = 180
        else:
         input = int(input)
        
        rba = request.form['rba']
        fed = request.form['fed']
        cpi = request.form['cpi']
        rba = float(rba)
        fed = float(fed)
        cpi = float(cpi)
        model=model_in_anz_rfr
        my_rf = randomforest(model,rba,fed,cpi)

        my_dict=predict(dates_df,last_sixty_anz,model_in_anz,input)

        path = '../static/data/images/anz_predict_graph.png'
        path2 = '../static/data/images/anz_tree.png'
        path3 = '../static/data/images/pred/anz_pred.png'
        return render_template('anz.html', href=path, dict=my_dict,hreftwo=path2,hrefthree=path3,dtree=my_rf,anz=anz_lstm,rf=anz_rf)

@app.route('/all.html', methods=('GET','POST'))
def predict_all():
    all_lstm = db.find_one({'model': 'LSTM','name': 'ALL'})
    all_rf = db.find_one({'model': 'RFR','name': 'ALL'})
    request_type = request.method
    if request_type == 'GET':
        m_dict=[{}]
        m_rfr=[{}]
        path1 = '../static/data/images/all_graph.png'
        path2 = '../static/data/images/all_tree.png'
        path3 = '../static/data/images/all.png'
        return render_template('all.html', href=path1,hreftwo=path2,hrefthree=path3,dict=m_dict,dtree=m_rfr,all=all_lstm,rf=all_rf)
    else:
        input = request.form['text']
        if input == "":
         input = 180
        else:
         input = int(input)
        
        rba = request.form['rba']
        fed = request.form['fed']
        cpi = request.form['cpi']
        rba = float(rba)
        fed = float(fed)
        cpi = float(cpi)
        model=model_in_all_rfr
        my_rf = randomforest(model,rba,fed,cpi)

        my_dict=predict(dates_df,last_sixty_all,model_in_all,input)

        path = '../static/data/images/all_predict_graph.png'
        path2 = '../static/data/images/all_tree.png'
        path3 = '../static/data/images/pred/all_pred.png'
        return render_template('all.html', href=path, dict=my_dict,hreftwo=path2,hrefthree=path3,dtree=my_rf,all=all_lstm,rf=all_rf)

@app.route('/sol.html', methods=('GET','POST'))
def predict_sol():
    sol_lstm = db.find_one({'model': 'LSTM','name': 'SOL'})
    sol_rf = db.find_one({'model': 'RFR','name': 'SOL'})
    request_type = request.method
    if request_type == 'GET':
        m_dict=[{}]
        m_rfr=[{}]
        path1 = '../static/data/images/sol_graph.png'
        path2 = '../static/data/images/sol_tree.png'
        path3 = '../static/data/images/sol.png'
        return render_template('sol.html', href=path1,hreftwo=path2,hrefthree=path3,dict=m_dict,dtree=m_rfr,sol=sol_lstm,rf=sol_rf)
    else:
        input = request.form['text']
        if input == "":
         input = 180
        else:
         input = int(input)
        
        rba = request.form['rba']
        fed = request.form['fed']
        cpi = request.form['cpi']
        rba = float(rba)
        fed = float(fed)
        cpi = float(cpi)
        model=model_in_sol_rfr
        my_rf = randomforest(model,rba,fed,cpi)

        my_dict=predict(dates_df,last_sixty_sol,model_in_sol,input)

        path = '../static/data/images/sol_predict_graph.png'
        path2 = '../static/data/images/sol_tree.png'
        path3 = '../static/data/images/pred/sol_pred.png'
        return render_template('sol.html', href=path, dict=my_dict,hreftwo=path2,hrefthree=path3,dtree=my_rf,sol=sol_lstm,rf=sol_rf)
    
@app.route('/pru.html', methods=('GET','POST'))
def predict_pru():
    pru_lstm = db.find_one({'model': 'LSTM','name': 'PRU'})
    pru_rf = db.find_one({'model': 'RFR','name': 'PRU'})
    request_type = request.method
    if request_type == 'GET':
        m_dict=[{}]
        m_rfr=[{}]
        path1 = '../static/data/images/pru_graph.png'
        path2 = '../static/data/images/pru_tree.png'
        path3 = '../static/data/images/pru.png'
        return render_template('pru.html', href=path1,hreftwo=path2,hrefthree=path3,dict=m_dict,dtree=m_rfr,pru=pru_lstm,rf=pru_rf)
    else:
        input = request.form['text']
        if input == "":
         input = 180
        else:
         input = int(input)
        
        rba = request.form['rba']
        fed = request.form['fed']
        cpi = request.form['cpi']
        rba = float(rba)
        fed = float(fed)
        cpi = float(cpi)
        model=model_in_pru_rfr
        my_rf = randomforest(model,rba,fed,cpi)

        my_dict=predict(dates_df,last_sixty_pru,model_in_pru,input)

        path = '../static/data/images/pru_predict_graph.png'
        path2 = '../static/data/images/pru_tree.png'
        path3 = '../static/data/images/pred/pru_pred.png'
        return render_template('pru.html', href=path, dict=my_dict,hreftwo=path2,hrefthree=path3,dtree=my_rf,pru=pru_lstm,rf=pru_rf)


# # Route that button will trigger the scrape function
# @app.route("/scrape")

# def scrape():
    # price_today = db_scrape.find_one(({'model': 'LSTM','name': 'LYC'}))

    # # Run the scrape function and save the results to a variable
    # var_data = scrape_shares.scrape_info()

    # # Update the Mongo database using update and upsert=True
    # mongo.db.collection.update_one({}, {"$set": var_data}, upsert=True)
   
    #Redirect back to home page
    return redirect("/", code=302)

if __name__ == "__main__":
    app.run(debug=True)