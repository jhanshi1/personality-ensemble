let radarChartInstance = null;

async function analyzeText() {

    const text = document.getElementById("inputText").value;
    const loading = document.getElementById("loading");

    if (text.length < 10) {
        alert("Please enter more text.");
        return;
    }

    loading.classList.remove("hidden");

    try {

        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: text })
        });

        const data = await response.json();

        displayPrimary(data.xgboost);
        displayOtherModels(data);

    } catch (error) {
        alert("Error connecting to backend.");
        console.error(error);
    }

    loading.classList.add("hidden");
}

function displayPrimary(xgbData) {

    const section = document.getElementById("primaryResult");
    const traitsDiv = document.getElementById("primaryTraits");

    section.classList.remove("hidden");

    const labels = ["OPN", "CON", "EXT", "AGR", "NEU"];
    const values = labels.map(trait => (xgbData[trait] * 100).toFixed(1));

    traitsDiv.innerHTML = "";

    labels.forEach((trait, i) => {
        traitsDiv.innerHTML += `
            <div class="trait-row">
                <span>${trait}</span>
                <span>${values[i]}%</span>
            </div>
        `;
    });

    const ctx = document.getElementById("radarChart").getContext("2d");

    if (radarChartInstance) {
        radarChartInstance.destroy();
    }

    radarChartInstance = new Chart(ctx, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [{
                label: "XGBoost Prediction (%)",
                data: values,
                backgroundColor: [
                    "#3b82f6",
                    "#06b6d4",
                    "#22c55e",
                    "#f59e0b",
                    "#ef4444"
                ],
                borderRadius: 10,
                barThickness: 60
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    labels: {
                        color: "#ffffff",
                        font: { size: 16 }
                    }
                }
            },
            scales: {
                x: {
                    ticks: {
                        color: "#ffffff",
                        font: { size: 16 }
                    },
                    grid: {
                        color: "rgba(255,255,255,0.05)"
                    }
                },
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        color: "#ffffff",
                        font: { size: 14 }
                    },
                    grid: {
                        color: "rgba(255,255,255,0.05)"
                    }
                }
            }
        }
    });
}

function displayOtherModels(data) {

    const comparison = document.getElementById("comparisonSection");
    const container = document.getElementById("otherModels");

    comparison.classList.remove("hidden");
    container.innerHTML = "";

    Object.entries(data).forEach(([model, traits]) => {

        if (model === "xgboost") return;

        let traitHTML = "";

        Object.entries(traits).forEach(([trait, score]) => {

            const percent = (score * 100).toFixed(1);

            traitHTML += `
                <div class="progress-row">
                    <span>${trait}</span>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width:${percent}%"></div>
                    </div>
                    <span>${percent}%</span>
                </div>
            `;
        });

        container.innerHTML += `
            <div class="model-card">
                <h3>${model.toUpperCase()}</h3>
                ${traitHTML}
            </div>
        `;
    });
}

function resetForm() {

    document.getElementById("inputText").value = "";

    document.getElementById("primaryResult").classList.add("hidden");
    document.getElementById("comparisonSection").classList.add("hidden");

    if (radarChartInstance) {
        radarChartInstance.destroy();
    }
}