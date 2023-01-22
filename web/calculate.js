function calculator() {
  var shape = extractRadioChoice("shape");
  var axis = extractRadioChoice("axis");
  var b = document.getElementById("b").value;
  var h = document.getElementById("h").value;
  var tf = document.getElementById("tf").value;
  var tw = document.getElementById("tw").value;
  var reb_dia = document.getElementById("reb_dia").value;
  var fcm = document.getElementById("fcm").value;
  var fcmgam = document.getElementById("gamma,fcm").value;
  var fa = document.getElementById("fa").value;
  var fagam = document.getElementById("gamma,fa").value;
  var fs = document.getElementById("fs").value;
  var fsgam = document.getElementById("gamma,fs").value;
  var conc_law = document.getElementById("conc_law").value;
  var depth = document.getElementById("depth").value;
  var reb_no = document.getElementById("no_reb").value;
  eel.calculate(
    fcm,
    fa,
    fs,
    fcmgam,
    fagam,
    fsgam,
    h,
    b,
    tw,
    tf,
    depth,
    reb_no,
    reb_dia,
    axis,
    shape
  )(cb);
}

cb = function (ret1) {
  console.log(ret1);
  plotCurves(ret1);
};
var form = document.getElementById("cal_btn2");
form.addEventListener("click", function (event) {
  event.preventDefault();
});
function extractRadioChoice(name) {
  var selected;
  var choices = document.getElementsByName(name);
  for (var i = 0; i < choices.length; i++) {
    if (choices[i].checked == true) {
      selected = choices[i].value;
    }
  }
  return selected;
}

function plotCurves(NM) {
  var trace1 = {
    x: NM[1],
    y: NM[0],
    type: "scatter",
  };
  var data = [trace1];

  Plotly.newPlot("figure", data);
}
