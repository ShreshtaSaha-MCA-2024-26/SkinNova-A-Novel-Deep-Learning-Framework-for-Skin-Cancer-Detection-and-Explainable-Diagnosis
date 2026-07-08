const chartFont = {
  family: "Poppins",
  size: 11,
  weight: "600"
};

const smallOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      display: true,
      labels: {
        boxWidth: 10,
        font: chartFont
      }
    }
  },
  scales: {
    x: {
      ticks: {
        font: chartFont
      },
      grid: {
        display: false
      }
    },
    y: {
      beginAtZero: true,
      ticks: {
        font: chartFont
      }
    }
  }
};

const sparkOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      display: false
    }
  },
  scales: {
    x: {
      display: false
    },
    y: {
      display: false
    }
  },
  elements: {
    point: {
      radius: 0
    }
  }
};

const miniAccuracy = document.getElementById("miniAccuracyChart");

if (miniAccuracy) {
  new Chart(miniAccuracy, {
    type: "line",
    data: {
      labels: ["1", "2", "3", "4", "5", "6", "7"],
      datasets: [{
        data: [82, 86, 88, 91, 93, 95, 96.8],
        borderColor: "#0d6efd",
        backgroundColor: "rgba(13, 110, 253, 0.12)",
        fill: true,
        tension: 0.4
      }]
    },
    options: sparkOptions
  });
}

const confidence = document.getElementById("confidenceChart");

if (confidence) {
  new Chart(confidence, {
    type: "doughnut",
    data: {
      labels: ["Confidence", "Remaining"],
      datasets: [{
        data: [94.2, 5.8],
        backgroundColor: ["#0d6efd", "#e5e7eb"],
        borderWidth: 0
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: "70%",
      plugins: {
        legend: {
          display: false
        }
      }
    }
  });
}

const training = document.getElementById("trainingChart");

if (training) {
  new Chart(training, {
    type: "line",
    data: {
      labels: ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"],
      datasets: [
        {
          label: "Accuracy",
          data: [78, 82, 86, 89, 91, 93, 95, 96.8],
          borderColor: "#0d6efd",
          backgroundColor: "rgba(13,110,253,.10)",
          tension: 0.4,
          fill: true
        },
        {
          label: "Loss",
          data: [60, 48, 39, 31, 25, 20, 17, 14],
          borderColor: "#ef4444",
          backgroundColor: "rgba(239,68,68,.08)",
          tension: 0.4,
          fill: true
        }
      ]
    },
    options: smallOptions
  });
}

const distribution = document.getElementById("distributionChart");

if (distribution) {
  new Chart(distribution, {
    type: "bar",
    data: {
      labels: ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"],
      datasets: [{
        label: "Images",
        data: [1113, 6705, 514, 327, 1099, 115, 142],
        backgroundColor: "#0d6efd",
        borderRadius: 6
      }]
    },
    options: smallOptions
  });
}

const risk = document.getElementById("riskChart");

if (risk) {
  new Chart(risk, {
    type: "doughnut",
    data: {
      labels: ["Low", "Medium", "High"],
      datasets: [{
        data: [58, 27, 15],
        backgroundColor: ["#22c55e", "#f59e0b", "#ef4444"],
        borderWidth: 0
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: "62%",
      plugins: {
        legend: {
          position: "bottom",
          labels: {
            font: chartFont,
            boxWidth: 10
          }
        }
      }
    }
  });
}

const classAccuracy = document.getElementById("classAccuracyChart");

if (classAccuracy) {
  new Chart(classAccuracy, {
    type: "bar",
    data: {
      labels: ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"],
      datasets: [{
        label: "Accuracy %",
        data: [94, 97, 95, 92, 96, 91, 93],
        backgroundColor: "#10b981",
        borderRadius: 6
      }]
    },
    options: smallOptions
  });
}