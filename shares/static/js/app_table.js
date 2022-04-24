// From data.js iterate through values
var tableData = data;
console.log(tableData);

//Appends table rows for all UFO sighting
var tbody = d3.select("tbody");
tableData.forEach((tableData) => {
    var row = tbody.append("tr");
    Object.entries(tableData).forEach(([key, value]) => {
      var cell = row.append("td");
      cell.text(value);
    });
});

//on filter button click/enter run filter
var form = d3.select(".form-control#datetime");
form.on("submit",runEnter);
var button = d3.select("#filter-btn.btn.btn-default");
button.on("click",runEnter);
// Event handler function for the input selected
function runEnter() {
    d3.event.preventDefault();
    // Select the input element and get value property
    var inputElement = d3.select("#datetime.form-control");
    var inputValue = inputElement.property("value");
    console.log(`date selected value: ${inputValue}`);
    //Filter based on date
    var filteredData = tableData.filter(t => t.datetime === String(inputValue));
    
  if (filteredData.length != 0 ) {
    Object.values(filteredData).forEach(value => console.log(value));
    //Repopulate table
    var tbody = d3.select("tbody");
    tbody.select("tr").remove();
    tbody.selectAll("td").remove();
      filteredData.forEach(filteredData => { 
         var row = tbody.append("tr");
         Object.values(filteredData).forEach(value => {
           var cell = row.append("td");
           cell.text(value);
          console.log("filter table event complete");
         });
     });
    } 
  else {
    alert("Not found! Select date between 1/1/2010 and 1/13/2010")
    console.log("Not found! Select bewteen 1/1/2010 and 1/13/2010")}
}