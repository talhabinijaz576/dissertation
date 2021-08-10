async function CheckUsername() {
    var username = document.getElementById("id_username").value;
    var url = window.location.href+"CheckUsername?username="+username;
    const response = await fetch(
        url, {headers : { 'Content-Type': 'application/json','Accept': 'application/json'}});
    var text = await response.json();
    var already_exists = String(text).includes("true");
    //console.log("Username already exists: " + already_exists);
    //document.getElementById("username_message").value = already_exists;
    return already_exists; 
}


async function CheckEmail() {
    var email = document.getElementById("id_email").value;
    if(email.length)
    var url = window.location.href+"CheckEmail?email="+email;
    const response = await fetch(
        url, {headers : { 'Content-Type': 'application/json','Accept': 'application/json'}});
    var text = await response.json();
    var already_exists = String(text).includes("true");
    //console.log("Email already exists: " + already_exists);
    //document.getElementById("username_message").value = already_exists;
    return already_exists; 
}

function onlyLettersAndNumber(str) {
  return str.match("^[A-Za-z0-9]+$") != null;
}

function isNumberKey(txt, evt)
{
    var charCode = (evt.which) ? evt.which : evt.keyCode;
    if (charCode != 43 && charCode > 31  && (charCode < 48 || charCode > 57))
      return false;
    // console.log(txt.value);
    if (charCode == 43 && (txt.value.indexOf('+') != -1 )) {
        return false;
    }

    return true;
}

function VerifyRegistrationForm() {

  //console.log("Verifying form");
  var errorLabel = document.getElementById("registerErrorLabel");

  var name = String(document.getElementById("id_name").value)
  if(name.length < 2){
    errorLabel.innerText = "Name cannot be empty";
    return false;
  }

  var phone = String(document.getElementById("id_phone").value)
  if(phone.length < 6){
    errorLabel.innerText = "Invalid phone number";
    return false;
  }

  var company_name = String(document.getElementById("id_company_name").value)
  if(company_name.length < 2){
    errorLabel.innerText = "Company name cannot be empty";
    return false;
  }

  var country = String(document.getElementById("id_country").value)
  if(country.length < 2){
    errorLabel.innerText = "Please select your country";
    return false;
  }

  var company_size = String(document.getElementById("id_company_size").value)
  if(company_size.length < 2){
    errorLabel.innerText = "Please select your company size";
    return false;
  }

  var business_role = String(document.getElementById("id_business_role").value)
  if(business_role.length < 2){
    errorLabel.innerText = "Please select your business role";
    return false;
  }


  var username = String(document.getElementById("id_username").value);
  if(username.length < 5 || username.length > 15) {
    errorLabel.innerText = "Username must be between 5 and 15 characters";
    return false;
  }
  //console.log(onlyLettersAndNumber(username));
  if(!onlyLettersAndNumber(username)){
    errorLabel.innerText = "Username can only contain letters and numbers";
    return false;
  }
  var username_exists = CheckUsername();
  username_exists.then(function(result) {
    if(result){
      //console.log("Username already exists");
      errorLabel.innerText = "Username already exists";
      return false;
    }
  });
  


  var email = document.getElementById("id_email").value;
  if(!email.includes(".") || !email.includes("@")) {
    errorLabel.innerText = "Invalid email";
    return false;
  }
  var email_exists = CheckEmail();
  email_exists.then(function(result) {
    result = false;
    if(result){
      //console.log("Email already exists");
      errorLabel.innerText = "Email already registered";
      return false;
    }
  });


  var password1 = document.getElementById("id_password").value;
  var password2 = document.getElementById("id_confirm_password").value;
  if(password1.length < 8 || password2.length > 20){
    errorLabel.innerText = "Password must be between 8 and 20 characters";
    return false;
  }
  if(password1!=password2){
    errorLabel.innerText = "Passwords do not match";
    return false;
  }

  var tac_box = document.getElementById("id_tac");
  if(!tac_box.checked){
    errorLabel.innerText = "Please accept the terms and conditions";
    return false;
  }

  return true;
}

function SubmitRegistrationForm() {
  var verified = VerifyRegistrationForm();
  if(verified == true){
    document.getElementById("RegistrationForm").submit();
  }
}
