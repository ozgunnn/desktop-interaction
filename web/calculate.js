var form = document.getElementById("form");
form.addEventListener("submit", function (event) {
  event.preventDefault();
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
  console.log(
    shape,
    axis,
    depth,
    b,
    h,
    tf,
    tw,
    reb_dia,
    reb_no,
    fcm,
    fa,
    fs,
    fcmgam,
    fagam,
    fsgam,
    conc_law
  );
  console.log(
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
    )
  );
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
