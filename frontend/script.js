

async function analyzeText() {

    const text = document.getElementById("inputText").value;
    const loading = document.getElementById("loading");
    const resultsSection = document.getElementById("resultsSection");

    if (text.length < 10) {
        alert("Please enter more text.");
        return;
    }

    loading.classList.remove("hidden");
    resultsSection.classList.add("hidden");

    try {

        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: text })
        });

        const data = await response.json();

        displayResults(data);

    } catch (error) {
        alert("Error connecting to backend.");
        console.error(error);
    }

    loading.classList.add("hidden");
}

function displayResults(data) {

    const resultsSection = document.getElementById("resultsSection");
    resultsSection.classList.remove("hidden");

    resultsSection.innerHTML = `
        <h2>Raw Prediction Output</h2>
        <pre style="background:#000; padding:20px; border-radius:10px; overflow:auto;">
${JSON.stringify(data, null, 2)}
        </pre>
    `;
}





function resetForm() {

    document.getElementById("inputText").value = "";
    document.getElementById("resultsSection").classList.add("hidden");
}
