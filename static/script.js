const sensitivitySlider = document.getElementById('sensitivity-slider');
const sensitivityValue = document.getElementById('sensitivity-value');
const thresholdSlider = document.getElementById('threshold-slider');
const thresholdValue = document.getElementById('threshold-value');
const thresholdDurationSlider = document.getElementById('threshold-duration-slider');
const thresholdDurationValue = document.getElementById('threshold-duration-value');
const startStopInput = document.getElementById('start-stop-input');
const testMicrophoneCheckbox = document.getElementById('test-microphone-checkbox');
const resultsList = document.getElementById('results');
const canvas = document.getElementById('waveformCanvas');
const ctx = canvas.getContext('2d');
const buffer = document.getElementById('buffer');
const loadingDialog = document.getElementById('loading-dialog');

let audioContext;
let microphone;
let scriptProcessor;
let gainNode;
let intervalId;
let intervalIdWf;
let sensitivity = 1.0;
let threshold = 50;
let duration = 2;
let holdClasses = [];

sensitivitySlider.addEventListener('input', updateSensitivity);
thresholdSlider.addEventListener('input', updateThresholdDisplay);
thresholdDurationSlider.addEventListener('input', updateThresholdDurationDisplay);
startStopInput.addEventListener('change', toggleProcessing);
testMicrophoneCheckbox.addEventListener('change', toggleMicrophoneTesting);

function updateSensitivity(event) {
    sensitivity = parseFloat(event.target.value);
    sensitivityValue.textContent = sensitivity;
    if (gainNode) {
        gainNode.gain.value = sensitivity;
    }
}

function updateThresholdDisplay(event) {
    threshold = parseInt(event.target.value, 10);
    thresholdValue.textContent = `${threshold} dB`;
}

function updateThresholdDurationDisplay(event) {
    duration = parseInt(event.target.value, 10);
    thresholdDurationValue.textContent = `${duration} sec`;
}

function toggleProcessing() {
    if (startStopInput.checked) {
        if (testMicrophoneCheckbox.checked) {
            testMicrophoneCheckbox.checked = false;
            stopTestingMicrophone();
        }
        disableSliders(true);
        showLoadingDialog();
        startProcessing();
    } else {
        disableSliders(false);
        showLoadingDialog();
        stopProcessing();
    }
}

function disableSliders(disable) {
    sensitivitySlider.disabled = disable;
    thresholdSlider.disabled = disable;
    thresholdDurationSlider.disabled = disable;
}

function startProcessing() {
    stopProcessing(); 

    buffer.style.display = 'block'; 
    loadingDialog.style.display = 'block'; 

    setTimeout(() => {
        intervalId = setInterval(fetchData, 1000); 
        intervalIdWf = setInterval(drawWaveformWrapper, 100);
        hideLoadingDialog(); 
    }, 3000); 
}

function stopProcessing() {

    buffer.style.display = 'block'; 
    loadingDialog.style.display = 'block'; 

    setTimeout(() => {
        clearInterval(intervalId);
        clearInterval(intervalIdWf);
        hideLoadingDialog(); 
    }, 3000);
}

function showLoadingDialog() {
    loadingDialog.style.display = 'block';
}

function hideLoadingDialog() {
    loadingDialog.style.display = 'none';
    buffer.style.display = 'none'; 
}

function fetchData() {
    const currentDate = new Date();
    const currentTime = currentDate.toLocaleString();

    fetch(`/classify_audio?sensitivity=${sensitivity}`)
        .then(response => response.json())
        .then(data => {
            const li = document.createElement('li');
            li.textContent = `${currentTime}, Class: ${data.class}, DB: ${data.db.toFixed(2)}`;

            if (data.db.toFixed(2) >= threshold) {
                holdClasses.push(data.class);
                if (holdClasses.length > duration) {
                    holdClasses.shift();
                }
            } else {
                holdClasses = [];
            }

            if (holdClasses.length === duration && holdClasses.every(elem => elem === data.class)) {
                li.textContent += ' - Ring';
                holdClasses = [];
            }
            resultsList.appendChild(li);
        })
        .catch(error => console.error('Error fetching data:', error));
}

function toggleMicrophoneTesting() {
    if (testMicrophoneCheckbox.checked) {
        clearInterval(intervalId);
        clearInterval(intervalIdWf);
        disableSliders(false);
        if (audioContext) {
            audioContext.close();
        }
        if (startStopInput.checked) {
            startStopInput.checked = false;
            stopProcessing();
        }

        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        gainNode = audioContext.createGain();
        gainNode.gain.value = sensitivity;

        navigator.mediaDevices.getUserMedia({ audio: true })
            .then(stream => {
                microphone = audioContext.createMediaStreamSource(stream);
                scriptProcessor = audioContext.createScriptProcessor(1024, 1, 1);

                microphone.connect(gainNode);
                gainNode.connect(scriptProcessor);
                scriptProcessor.connect(audioContext.destination);

                scriptProcessor.onaudioprocess = (event) => {
                    const inputBuffer = event.inputBuffer;
                    const outputBuffer = event.outputBuffer;

                    for (let channel = 0; channel < inputBuffer.numberOfChannels; channel++) {
                        const inputData = inputBuffer.getChannelData(channel);
                        const outputData = outputBuffer.getChannelData(channel);

                        let rms = 0;
                        for (let sample = 0; sample < inputBuffer.length; sample++) {
                            outputData[sample] = inputData[sample];
                            rms += inputData[sample] ** 2;
                        }
                        rms = Math.sqrt(rms / inputBuffer.length);
                        const db = 20 * Math.log10(rms);
                    }
                };
            })
            .catch(error => console.error('Error accessing microphone:', error));
    } else {
        stopTestingMicrophone();
    }
}

function stopTestingMicrophone() {
    if (scriptProcessor) {
        scriptProcessor.disconnect();
    }
    if (gainNode) {
        gainNode.disconnect();
    }
    if (microphone) {
        microphone.disconnect();
    }
    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }
}

function drawWaveformWrapper() {
    fetch(`/classify_audio?sensitivity=${sensitivity}`)
        .then(response => response.json())
        .then(data => {
            if (data.waveform) {
                drawWaveform(data.waveform);
            }
        })
        .catch(error => console.error('Error fetching waveform:', error));
}

function drawWaveform(waveform) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.lineWidth = 2;
    ctx.strokeStyle = '#000';

    const scale = canvas.width / waveform.length;
    const offset = canvas.height / 2;

    ctx.beginPath();
    ctx.moveTo(0, offset - waveform[0] * offset);
    for (let i = 1; i < waveform.length; i++) {
        ctx.lineTo(i * scale, offset - waveform[i] * offset);
    }
    ctx.stroke();
}
