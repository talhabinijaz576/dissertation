
var deleteConfirmationModal = document.getElementById("deleteConfirmationModal");
var deleteConfirmationClose = document.getElementById("deleteConfirmationClose");

var adddeveloperTrigger = document.getElementById("adddeveloperTrigger");
var editdeveloperModal = document.getElementById("editdeveloperModal");
var editdeveloperClose = document.getElementById("editdeveloperClose");
var errorLabel = document.getElementById("editdeveloperErrorLabel");


editdeveloperClose.onclick = function() {
  editdeveloperModal.style.display = "none";
  ResetForm();
}

adddeveloperTrigger.onclick = function() {
  editdeveloperModal.style.display = "block";
}


deleteConfirmationClose.onclick = function() {
  deleteConfirmationModal.style.display = "none";
}



window.onclick = function(event) {

  if (event.target === deleteConfirmationModal) {
    deleteConfirmationModal.style.display = "none";
  }

}

function TogglePassword(element_id){
  var element = document.getElementById(element_id);
  if(element.type=="text"){
    element.type="password";
  }else{
    element.type="text";
  }
}

function ConfirmDelete(key, value){
  document.getElementById("deleteName").innerText = value;
  var input = document.getElementById("deleteConfirmationInput");
  input.name=key;
  input.value = value;
  deleteConfirmationModal.style.display = "block";
}

function VerifyForm(){


  var password1 = document.getElementById("password1").value;
  var password2 = document.getElementById("password2").value;
  var developer_sent = document.getElementById("developer_sent").value;
  //console.log("developer sent value: " + developer_sent + " "+ developer_sent.length);

  if(password1.length < 8 && (developer_sent.length==0 || password1.length > 0 || password2.length > 0)){
    errorLabel.innerText = "Password must be atleast 8 characters"
    return false;
  }

  if(password1 != password2){
    errorLabel.innerText = "Passwords do not match"
    return false;
  }

  return true;
}

function SubmitdeveloperForm(){
  var verified = VerifyForm();
  if(verified){
    errorLabel.innerText = ""
    document.getElementById("editdeveloperForm").submit();
  }
}

function ResetForm(){
  console.log("Reseting Form");
  errorLabel.innerText = "";
  document.getElementById("editdeveloperModalTitle").innerText  = "Create developer";
  document.getElementById("update_process").innerText = "Create";
  document.getElementById("id_email_div").classList.remove("is-filled");
  document.getElementById("id_password_div").classList.remove("is-filled");
  document.getElementById("id_confirm_password_div").classList.remove("is-filled");
  document.getElementById("developer_email").value = "";
  document.getElementById("password1").value = "";
  document.getElementById("password2").value = "";
  document.getElementById("developer_sent").value = "";
  document.getElementById("developerName").innerText = document.getElementById("new_developer_name").value;
  document.getElementById("access_to_code").setAttribute("checked", "")
  document.getElementById("access_to_assets").setAttribute("checked", "")
  document.getElementById("is_project_admin").removeAttribute("checked")
}