// Global variable for data coming from mongo database through flask app.py
var mdata
console.log("arrived")
// Function to load the result
d3.select("#submit").on("click",loadResults);

function loadResults(){
  d3.json("/mdata").then((importedData) => {
    mdata = importedData.shares
  
 Object.entries(mdata).forEach(([key, value]) =>{
    console.log("data values: "+mdata)
  
  var option_co = d3.select("#sel1").property("value");
  console.log(`option co found value ${option_co}`);
  option_co = option_co.toString()

  var option_model = d3.select("#radio1").property("value");
  console.log(`option model found value ${option_model}`);
  option_model = option_model.toString()

  d3.select("#sample-metadata").html("");
  for (let i = 0; i < mdata.length; i++) {
    let row = mdata[i];

    if (row.Name == option_co & row.Model == option_model) {
      var rowActual = parseInt(row.Actual);
      var rowPredict = parseInt(row.Predicted);
      var rowDiff = parseInt(row.difference);
     console.log(rowActual,rowPredict,rowDiff)

     
     //Appends html metadate for that induvidual/row selected
     var dbody = d3.select("div#sample-metadata.panel-body").append("table").attr("class","table-responsive").append("tbody").append("tr");
     Object.entries(row).forEach(([key, value]) => {
       console.log(`Key: ${key} and Value: ${value}`);
       var cell = dbody.append("tr").append("td");
       cell.text(` ${key.toUpperCase()} : "${value}" `)
       });

     
    };//close if
  };//close for loop

  })});//close Then and forEach line 9 & 12

}; //close function line 8