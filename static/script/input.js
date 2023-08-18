const fileInput = document.getElementById("fileInput");
const selectedFileName = document.getElementById("selectedFileName");
const uploadButton = document.getElementById("uploadButton");
const fileUploadForm = document.getElementById("fileUploadForm");
const uploadResultSuccess = document.getElementById('uploadResultSuccess');
const uploadResultFailure = document.getElementById("uploadResultFailure");
const predictButton = document.getElementById("predictButton");
const predictionProgress = document.getElementById("predictionProgress");
const predictionMessageSuccess = document.getElementById("predictionMessageSuccess")
const predictionMessageFailure = document.getElementById("predictionMessageFailure")
const downloadReport = document.getElementById("downloadReport")


fileInput.addEventListener("change", function () {
    // Get the selected file and display its name
    const file = fileInput.files[0];
    if (file) {
        selectedFileName.textContent = file.name;
    } else {
        selectedFileName.textContent = "";
    }

    uploadResultFailure.textContent = "";
    uploadResultSuccess.textContent = "";
    predictionMessageSuccess.textContent = "";
    predictionMessageFailure.textContent = "";
    downloadReport.textContent = "";

});

uploadButton.addEventListener("click", function (event) {
    event.preventDefault();
    // Implement your file upload functionality here
    // You can use JavaScript or a backend server to process the uploaded data
    const file = fileInput.files[0];
    if (file) {
        // check file extension
        extension = getFileExtension(file.name)
        if (extension != 'xls' && extension != 'xlsx') {
            uploadResultFailure.textContent = "File extension not correct. Please select .xls or .xlsx file only";
            return
        }


        const formData = new FormData();
        formData.append("file", file);
        fetch("/upload", {
            method: "POST",
            body: formData,
        })
            .then((response) => {
                if (!response.ok) {
                    throw new Error("Network response was not ok");
                }
                return response.json();
            })
            .then((data) => {
                // Handle the response from the server (if needed)
                if (data['status'] == 200) {
                    uploadResultSuccess.textContent = "Upload successful. Now you can predict";
                    predictButton.disabled = false; // enable the predict button
                }
                else if (data['status'] == 400) {
                    uploadResultFailure.textContent = "File upload failed";
                }
            })
            .catch((error) => {
                // Handle any errors that occurred during the upload process
                console.error("Error uploading the file:", error);
            });
    }

    // Reset the file input and selected file name display
    fileUploadForm.reset();
    selectedFileName.textContent = "";
});

//Listener for predict
predictButton.addEventListener("click", function (event) {
    // reset the uploadresult text
    uploadResultFailure.textContent = "";
    uploadResultSuccess.textContent = "";

    //disable the upload and select button
    uploadButton.disabled = true
    fileInput.disabled = true
    predictButton.disabled = true
    predictionProgress.textContent = "Prediction in progress ...."

    // call the api for prediction
    fetch("/predict")
        .then((response) => {
            if (!response.ok) {
                throw new Error("Network response was not ok");
            }
            return response.json();
        })
        .then((data) => {
            if (data['status'] == 200) {
                predictionMessageSuccess.textContent = data['message']
                downloadReport.textContent = "Download Full Report"
            }
            else if (data['status'] == 400) {
                predictionMessageFailure.textContent = data['message']
            }
            uploadButton.disabled = false
            fileInput.disabled = false
            predictionProgress.textContent = ""
        })

});

function getFileExtension(filename) {
    // Use a regular expression to extract the file extension
    const extension = filename.match(/\.[0-9a-z]+$/i);

    // If the regular expression found a match, return the extension (without the dot)
    if (extension) {
        return extension[0].substring(1);
    }

    // If no extension was found, return empthy string
    return "";
}

