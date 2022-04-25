/// data in a list of many dicts/rows
console.log(data)

function makeResponsive() {
  var svgArea = d3.select("body").select("svg");
  if (!svgArea.empty()) {
    svgArea.remove();
  }
  var svgWidth = window.innerWidth;
  var svgHeight = window.innerHeight;

  // set the dimensions and margins of the graph
  var margin = { top: 5, right: 450, bottom: 100, left: 80 },
    width = svgWidth - margin.left - margin.right,
    height = svgHeight - margin.top - margin.bottom;

  // append the svg object to the body of the page and group element
  var svg = d3
    .select("#scatter")
    .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
    .append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

  //Read the data
  //d3.csv("../data/cba.csv").then( function(data) {
  d3.json("/cba_data")
    .then(data => {
      Object.entries(data).forEach(([key, value]) => {
        console.log(key, value);
      });

      var chosenXAxis = "Volume"; //default
      var chosenYAxis = "Close"; //default
      // Add X axis

      if (chosenXAxis === "Volume") {
        var xmin = 150;
        var xmax = 25000000;
      }
      if (chosenXAxis === "RBA") {
        var xmin = -0.05;
        var xmax = 7.5;
      }
      if (chosenXAxis === "FED") {
        var xmin = -0.05;
        var xmax = 7;
      }

      var x = d3
        .scaleLinear()
        .domain([xmin, xmax])
        // var x = d3.scaleLinear()
        // .domain([d3.min(data, d => d[chosenXAxis]) * 0.97,
        //  d3.max(data, d => d[chosenXAxis]) * 1.02])
        .range([0, width]);
      svg
        .append("g")
        .attr("transform", "translate(0," + height + ")")
        .attr("class", "x_axislabels")
        .call(d3.axisBottom(x));

      if (chosenYAxis === "Open") {
        var ymin = 20;
        var ymax = 115;
      }
      if (chosenYAxis === "High") {
        var ymin = 20;
        var ymax = 115;
      }
      if (chosenYAxis === "Close") {
        var ymin = 20;
        var ymax = 115;
      }

      // Add Y axis
      var y = d3
        .scaleLinear()
        // .domain([d3.min(data, d => d[chosenYAxis]) * 0.94,
        //  d3.max(data, d => d[chosenYAxis]) * 1.05])
        .domain([ymin, ymax])
        .range([height, 0]);
      svg.append("g").attr("class", "y_axislabels").call(d3.axisLeft(y));

      // Add a scale for bubble size
      const z = d3.scaleLinear().domain([20, 140]).range([4, 10]);

      // Add a scale for bubble color
      const myColor = d3
        .scaleOrdinal()
        // .range(d3.schemeSet3);
        .range(d3.schemePaired);

      // -1- Create a tooltip div that is hidden by default:
      const tooltip = d3
        .select("#scatter")
        .append("div")
        .style("opacity", 0)
        .attr("class", "tooltip")
        .style("background-color", "black")
        .style("border-radius", "5px")
        .style("padding", "10px")
        .style("color", "white");

      // -2- Create 3 functions to show / update (when mouse move but stay on same circle) / hide the tooltip

      const showTooltip = function (event, d) {
        tooltip.transition().duration(200);
        chosenYAxis = d3.select("text.ylabels.active").attr("value");
        tooltip
          .style("opacity", 1)
          .html(
            `${d.Date}<br>${chosenXAxis.toUpperCase()}: ${
              d[chosenXAxis]
            }<br>${chosenYAxis.toUpperCase()}: ${d[chosenYAxis]}`
          )
          .style("left", event.x / 2 + "px")
          .style("top", event.y / 2 + 30 + "px");
      };
      const moveTooltip = function (event, d) {
        tooltip
          .style("left", event.x / 2 + "px")
          .style("top", event.y / 2 + 30 + "px");
      };
      const hideTooltip = function (event, d) {
        tooltip.transition().duration(200).style("opacity", 0);
      };

      // Add dots

      var circlesGroup = svg
        .append("g")
        .attr("class", "node_wrapper")
        .selectAll("dot")
        .data(data)
        .join("g")
        .attr("class", "bubble_wrapper")
        .append("circle")
        .attr("class", "bubbles")
        .attr("cx", (d) => x(d[chosenXAxis]))
        .attr("cy", (d) => y(d[chosenYAxis]))
        .attr("r", (d) => z(d.Close))
        .style("fill", (d) => myColor(d.Date.slice(-4)))
        .on("mouseover", showTooltip)
        .on("mousemove", moveTooltip)
        .on("mouseleave", hideTooltip);

      d3.selectAll(".bubble_wrapper")
        .data(data)
        //.append("text").text(d => d.Date)
        .attr("text-anchor", "middle")
        .attr("x", (d) => x(d[chosenXAxis]))
        .attr("y", (d) => y(d[chosenYAxis]));

      // Create Extra y-axis click labels
      var ylabelsGroup = svg
        .append("g")
        .attr("class", "y_options")
        .attr("transform", `translate(${width - width - 95}, ${height - 250})`);

      var openLabel = ylabelsGroup
        .append("text")
        .attr("x", 40)
        .attr("y", -40)
        .attr("class", "ylabels")
        .attr("value", "Open") // value to grab for event listener
        .classed("active", true)
        .text("Open");

      var highLabel = ylabelsGroup
        .append("text")
        .attr("x", 40)
        .attr("y", -20)
        .attr("class", "ylabels")
        .attr("value", "High") // value to grab for event listener
        .classed("active", false)
        .classed("inactive", true)
        .text("High");

      var closeLabel = ylabelsGroup
        .append("text")
        .attr("x", 40)
        .attr("y", 0)
        .attr("class", "ylabels")
        .attr("value", "Close") // value to grab for event listener
        .classed("active", false)
        .classed("inactive", true)
        .text("Close");

      // Create Extra x-axis click labels
      var xlabelsGroup = svg
        .append("g")
        .attr("class", "x_options")
        .attr("transform", `translate(${width / 2}, ${height + 10})`);

      var volumeLabel = xlabelsGroup
        .append("text")
        .attr("x", 0)
        .attr("y", 30)
        .attr("class", "xlabels")
        .attr("value", "Volume") // value to grab for event listener
        .classed("active", true)
        .text("Volume");

      var rbaLabel = xlabelsGroup
        .append("text")
        .attr("x", 0)
        .attr("y", 50)
        .attr("class", "xlabels")
        .attr("value", "RBA") // value to grab for event listener
        .classed("inactive", true)
        .text("RBA Interest Rate");

      var fedLabel = xlabelsGroup
        .append("text")
        .attr("x", 0)
        .attr("y", 70)
        .attr("class", "xlabels")
        .attr("value", "FED") // value to grab for event listener
        .classed("inactive", true)
        .text("FED Interest Rate");

      // $$$ Y AXIS Event listener Click $$$
      d3.selectAll("text.ylabels").on("click", function () {
        var value = d3.select(this).attr("value");
        var chosenYAxis = value;
        d3.select(this).classed("active", true);
        d3.select(this).classed("inactive", false);

        console.log("Values currently selected");
        console.log(chosenXAxis);
        console.log(chosenYAxis);

        d3.selectAll("g.x_axislabels").remove();
        d3.selectAll("g.y_axislabels").remove();
        d3.selectAll("g.node_wrapper").remove().transition().duration(9000);

        if (chosenXAxis === "Volume") {
          var xmin = 150;
          var xmax = 25000000;
        }
        if (chosenXAxis === "RBA") {
          var xmin = -0.05;
          var xmax = 7.5;
        }
        if (chosenXAxis === "FED") {
          var xmin = -0.05;
          var xmax = 7;
        }

        var x = d3.scaleLinear().domain([xmin, xmax]).range([0, width]);
        svg
          .append("g")
          .attr("transform", "translate(0," + height + ")")
          .attr("class", "x_axislabels")
          .call(d3.axisBottom(x));

        if (chosenYAxis === "Open") {
          var ymin = 20;
          var ymax = 115;
        }
        if (chosenYAxis === "High") {
          var ymin = 20;
          var ymax = 115;
        }
        if (chosenYAxis === "Close") {
          var ymin = 20;
          var ymax = 115;
        }

        var y = d3.scaleLinear().domain([ymin, ymax]).range([height, 0]);
        svg
          .append("g")
          .transition()
          .duration(600)
          .attr("class", "y_axislabels")
          .call(d3.axisLeft(y));

        var circlesGroup = svg
          .append("g")
          .attr("class", "node_wrapper")
          .selectAll("dot")
          .data(data)
          .join("g")
          .attr("class", "bubble_wrapper")
          .append("circle")
          .attr("class", "bubbles")
          .attr("cx", (d) => x(d[chosenXAxis]))
          .attr("cy", (d) => y(d[chosenYAxis]))
          .attr("r", (d) => z(d.Close))
          .style("fill", (d) => myColor(d.Date.slice(-4)))
          // -3- Trigger the functions
          .on("mouseover", showTooltip)
          .on("mousemove", moveTooltip)
          .on("mouseleave", hideTooltip);
        d3.selectAll(".bubble_wrapper")
          .data(data)
          // .append("text").text(d => d.Date)
          .attr("text-anchor", "middle")
          .attr("x", (d) => x(d[chosenXAxis]))
          .attr("y", (d) => y(d[chosenYAxis]))
          .transition()
          .duration(4000);

        // Unselected option classes to change bold text off
        if (chosenYAxis !== "Open") {
          openLabel.classed("active", false).classed("inactive", true);
        }
        if (chosenYAxis !== "High") {
          highLabel.classed("active", false).classed("inactive", true);
        }
        if (chosenYAxis !== "Close") {
          closeLabel.classed("active", false).classed("inactive", true);
        }
      });

      // $$$ X AXIS Event listener Click $$$
      d3.selectAll("text.xlabels").on("click", function () {
        var value = d3.select(this).attr("value");
        chosenXAxis = value;
        chosenYAxis = d3.select("text.ylabels.active").attr("value");
        d3.select(this).classed("active", true);
        d3.select(this).classed("inactive", false);

        console.log("Values currently selected");
        console.log(chosenXAxis);
        console.log(chosenYAxis);

        d3.selectAll("g.x_axislabels").remove();
        d3.selectAll("g.y_axislabels").remove();
        d3.selectAll("g.node_wrapper").remove().transition().duration(9000);

        if (chosenXAxis === "Volume") {
          var xmin = 150;
          var xmax = 25000000;
        }
        if (chosenXAxis === "RBA") {
          var xmin = -0.08;
          var xmax = 7.5;
        }
        if (chosenXAxis === "FED") {
          var xmin = -0.05;
          var xmax = 7.0;
        }

        //update x-Axis
        var x = d3.scaleLinear().domain([xmin, xmax]).range([0, width]);
        svg
          .append("g")
          .transition()
          .duration(400)
          .attr("transform", "translate(0," + height + ")")
          .attr("class", "x_axislabels")
          .call(d3.axisBottom(x));

        if (chosenYAxis === "Open") {
          var ymin = 20;
          var ymax = 115;
        }
        if (chosenYAxis === "High") {
          var ymin = 20;
          var ymax = 115;
        }
        if (chosenYAxis === "Close") {
          var ymin = 20;
          var ymax = 115;
        }

        var y = d3.scaleLinear().domain([ymin, ymax]).range([height, 0]);
        svg.append("g").attr("class", "y_axislabels").call(d3.axisLeft(y));

        var circlesGroup = svg
          .append("g")
          .attr("class", "node_wrapper")
          .selectAll("dot")
          .data(data)
          .join("g")
          .attr("class", "bubble_wrapper")
          .append("circle")
          .attr("class", "bubbles")
          .attr("cx", (d) => x(d[chosenXAxis]))
          .attr("cy", (d) => y(d[chosenYAxis]))
          .attr("r", (d) => z(d.Close))
          .style("fill", (d) => myColor(d.Date.slice(-4)))
          // -3- Trigger tooltip functions
          .on("mouseover", showTooltip)
          .on("mousemove", moveTooltip)
          .on("mouseleave", hideTooltip);

        d3.selectAll(".bubble_wrapper")
          .data(data)
          // .append("text").text(d => d.Date)
          .attr("text-anchor", "middle")
          .attr("x", (d) => x(d[chosenXAxis]))
          .attr("y", (d) => y(d[chosenYAxis]))
          .transition()
          .duration(9000);

        // unselected option classes to change bold text off
        if (chosenXAxis !== "Volume") {
          volumeLabel.classed("active", false).classed("inactive", true);
        }
        if (chosenXAxis !== "RBA") {
          rbaLabel.classed("active", false).classed("inactive", true);
        }
        if (chosenXAxis !== "FED") {
          fedLabel.classed("active", false).classed("inactive", true);
        }
      }); // on click

      // ****** Extra labels end *******

      //brackets to close then function and catch then errors
    })
    .catch(function (error) {
      console.log(error);
    });

  //brackets to close responsive function
}

// When the browser loads, makeResponsive() is called.
makeResponsive();

// When the browser window is resized, makeResponsive() is called.
d3.select(window).on("resize", makeResponsive);
