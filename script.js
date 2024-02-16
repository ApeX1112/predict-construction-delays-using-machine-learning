document.getElementById('predictionForm').onsubmit = function(event) {
    event.preventDefault();
    const input1 = document.getElementById('input1').value;
    const input3 = document.getElementById('input3').value;
    const input4 = document.getElementById('input4').value;
    const input5 = document.getElementById('input5').value;
    const sliderInput = document.getElementById('sliderInput').value;
    const sliderInput1 = document.getElementById('sliderInput1').value;
    const sliderInput2 = document.getElementById('sliderInput2').value;
    const sliderInput3 = document.getElementById('sliderInput3').value;
    const sliderInput4 = document.getElementById('sliderInput4').value;
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            input1: input1,
            
            input3: input3,
            input4: input4,
            input5: input5,
            sliderInput: sliderInput,
            sliderInput1: sliderInput1,
            sliderInput2: sliderInput2,
            sliderInput3: sliderInput3,
            sliderInput4: sliderInput4

        }),
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('predictionResult').textContent = 'Prediction: ' + data;
    })
    .catch((error) => {
        console.error('Error:', error);
    });
};

// Update slider value display in real-time
document.getElementById('sliderInput').oninput = function(event) {
    document.getElementById('sliderValue').textContent = event.target.value;
};

document.getElementById('sliderInput1').oninput = function(event) {
    document.getElementById('sliderValue1').textContent = event.target.value;
};

document.getElementById('sliderInput2').oninput = function(event) {
    document.getElementById('sliderValue2').textContent = event.target.value;
};

document.getElementById('sliderInput3').oninput = function(event) {
    document.getElementById('sliderValue3').textContent = event.target.value;
};

document.getElementById('sliderInput4').oninput = function(event) {
    document.getElementById('sliderValue4').textContent = event.target.value;
};