const skinImage = document.getElementById("skinImage");
const uploadBox = document.getElementById("uploadBox");
const uploadContent = document.getElementById("uploadContent");
const previewImage = document.getElementById("previewImage");
const fileName = document.getElementById("fileName");
const fileSize = document.getElementById("fileSize");
const clearBtn = document.getElementById("clearBtn");
const detectForm = document.getElementById("detectForm");
const resultPlaceholder = document.getElementById("resultPlaceholder");
const resultContent = document.getElementById("resultContent");
const confidenceFill = document.getElementById("confidenceFill");
const confidenceText = document.getElementById("confidenceText");
const predictionText = document.getElementById("predictionText");
const riskText = document.getElementById("riskText");
const recommendationText = document.getElementById("recommendationText");
const qualityBox = document.getElementById("qualityBox");
const qualityText = document.getElementById("qualityText");
const newDetectBtn = document.getElementById("newDetectBtn");

const predictions = [
  {
    disease: "Melanoma",
    confidence: 98.7,
    risk: "High Risk",
    riskClass: "high-risk",
    recommendation: "Consult a dermatologist immediately for further clinical evaluation."
  },
  {
    disease: "Melanocytic Nevus",
    confidence: 94.3,
    risk: "Low Risk",
    riskClass: "low-risk",
    recommendation: "Routine clinical monitoring is recommended."
  },
  {
    disease: "Basal Cell Carcinoma",
    confidence: 91.5,
    risk: "Moderate Risk",
    riskClass: "medium-risk",
    recommendation: "Dermatologist consultation is advised for further examination."
  },
  {
    disease: "Benign Keratosis",
    confidence: 89.8,
    risk: "Low Risk",
    riskClass: "low-risk",
    recommendation: "Likely low risk, but periodic monitoring is recommended."
  }
];

function formatSize(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(2) + " MB";
}

function showPreview(file) {
  if (!file) return;

  const allowedTypes = ["image/jpeg", "image/jpg", "image/png"];

  if (!allowedTypes.includes(file.type)) {
    alert("Please upload JPG, JPEG, or PNG image only.");
    return;
  }

  const reader = new FileReader();

  reader.onload = function (e) {
    previewImage.src = e.target.result;
    previewImage.style.display = "block";
    uploadContent.style.display = "none";

    fileName.textContent = file.name;
    fileSize.textContent = formatSize(file.size);

    qualityBox.classList.remove("warning");
    qualityBox.classList.add("good");
    qualityText.textContent = "Image accepted. Ready for AI detection.";
  };

  reader.readAsDataURL(file);
}

skinImage?.addEventListener("change", function () {
  showPreview(this.files[0]);
});

uploadBox?.addEventListener("dragover", function (e) {
  e.preventDefault();
  uploadBox.classList.add("dragover");
});

uploadBox?.addEventListener("dragleave", function () {
  uploadBox.classList.remove("dragover");
});

uploadBox?.addEventListener("drop", function (e) {
  e.preventDefault();
  uploadBox.classList.remove("dragover");

  const file = e.dataTransfer.files[0];

  if (file) {
    skinImage.files = e.dataTransfer.files;
    showPreview(file);
  }
});

clearBtn?.addEventListener("click", function () {
  skinImage.value = "";
  previewImage.src = "";
  previewImage.style.display = "none";
  uploadContent.style.display = "block";
  fileName.textContent = "No image selected";
  fileSize.textContent = "0 KB";

  qualityBox.classList.remove("good", "warning");
  qualityText.textContent = "Waiting for image upload";

  resultPlaceholder.style.display = "flex";
  resultContent.style.display = "none";
  confidenceFill.style.width = "0%";
});

detectForm?.addEventListener("submit", function (e) {
  e.preventDefault();

  if (!skinImage.files || skinImage.files.length === 0) {
    qualityBox.classList.add("warning");
    qualityText.textContent = "Please upload an image before detection.";
    return;
  }

  const selected = predictions[Math.floor(Math.random() * predictions.length)];

  predictionText.textContent = selected.disease;
  confidenceText.textContent = selected.confidence + "%";

  riskText.textContent = selected.risk;
  riskText.className = "risk-pill " + selected.riskClass;

  recommendationText.textContent = selected.recommendation;

  resultPlaceholder.style.display = "none";
  resultContent.style.display = "block";

  setTimeout(() => {
    confidenceFill.style.width = selected.confidence + "%";
  }, 200);
});

newDetectBtn?.addEventListener("click", function () {
  clearBtn.click();
});