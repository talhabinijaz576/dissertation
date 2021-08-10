

var settingsForm = document.getElementById("settingsForm");
var errorLabel = document.getElementById("editSettingsErrorLabel");
var timezoneDiv = document.getElementById("timezoneDiv");
var timezoneInput = document.getElementById("timezoneInput");
var timezoneDropdown = document.getElementById("timezoneListDropdown");


window.onclick = function(event) {
  if (event.target != timezoneInput) {
    timezoneDropdown.classList.remove('show');
  }
}

function ToggleDropdownList(element_id) {
  document.getElementById(element_id).classList.toggle("show");
}
function ActivateDropdownList(element_id) {
  document.getElementById(element_id).classList.add("show");
}

function SetTimezone(timezone){
  timezoneInput.classList.remove("show");
  timezoneDiv.classList.add("is-filled");
  timezoneInput.setAttribute("value", timezone);
  timezoneInput.value = timezone;
}

function filterFunction() {
  ActivateDropdownList('timezoneListDropdown');
  var input, filter, ul, li, a, i;
  input = timezoneInput;
  filter = input.value.toUpperCase();
  div = timezoneDropdown;
  a = div.getElementsByTagName("a");
  for (i = 0; i < a.length; i++) {
    txtValue = a[i].textContent || a[i].innerText;
    if (txtValue.toUpperCase().indexOf(filter) > -1) {
      a[i].style.display = "";
    } else {
      a[i].style.display = "none";
    }
  }
}

function VerifySettings(timezones){
  var verified = VerifyTimezone(timezones);
  return verified;
}

function VerifyTimezone(timezones){
  var timezone = timezoneInput.value
  var verified = timezones.includes(timezone) && timezone.length>=3;
  if(!verified){
    errorLabel.innerText = "Please select a timezone from the list";
  }
  return verified;

}

function SubmitSettingsForm(timezones){
  var verified = VerifySettings(timezones);
  if(verified){
    errorLabel.innerText = "";
    settingsForm.submit();
  }
}


showNotification('top','center');
