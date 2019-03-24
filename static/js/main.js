
var video = document.querySelector("#videoElement");

if (navigator.mediaDevices.getUserMedia) {
  navigator.mediaDevices.getUserMedia({ video: true })
    .then(function (stream) {
      video.srcObject = stream;
    })
    .catch(function (error) {
      console.log("Something went wrong!");
    });
}

function snap()
{
    const track = mediaStream.getVideoTracks()[0];
    var imageCapture = new ImageCapture(track);
    document.getElementById("form").submit();
};