function handleUpload() {
  const file = document.getElementById("imageInput").files[0];
  if (!file) {
    alert("Please select an image first.");
    return;
  }

  const reader = new FileReader();
  reader.onload = function (e) {
    localStorage.setItem("previewURL", e.target.result); // base64 image
    window.location.href = "result.html";
  };
  reader.readAsDataURL(file);
}
