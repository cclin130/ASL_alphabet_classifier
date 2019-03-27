//code for ongoing video display
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

//run snap when spacebar is pressed
document.body.onkeyup = function(e){
    if(e.keyCode == 32){
        snap()
    }
}

//deifne a function for when button is pressed
function snap()
{
    var blob;

    if (navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: true })
        .then(mediaStream => {
            //get one screen from the video track (i.e. a photo)
            const track = mediaStream.getVideoTracks()[0];
            var imageCapture = new ImageCapture(track);

            //take the photo of the single screen capture
            imageCapture.takePhoto()
                .then(blob => {
                    //once photo is taken, create a formData object to store it
                    video.src = URL.createObjectURL(blob);
                    video.onload = () => { URL.revokeObjectURL(this.src); }

                    var formData = new FormData();
                    formData.append("image", blob);

                    //send an XMLHttpRequest to flask backend
                    var request = new XMLHttpRequest();
                    request.open("POST", "http://localhost:5000/");
                    request.withCredentials = true;

                    //define an onload function for when backend returns response
                    request.onload = function (){
                        if (request.readyState === request.DONE){
                            if (request.status === 200){
                                console.log(request.response);
                                var result = document.getElementById("output");
                                var result_display = document.getElementById("output_display");
                                result.innerHTML = request.response;
                                result_display.style.visibility = "visible";
                            }
                        }                
                    }

                request.send(formData);

              })
              .catch(error => console.error('takePhoto() error:', error));
        })
        .catch(function (error) {
            console.log("Something went wrong!");
            console.log(error);
        })
    }
};