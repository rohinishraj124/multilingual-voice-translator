let mediaRecorder;
let audioChunks = [];
let isRecording = false;
let startTime;
let timerInterval;

const recordBtn = document.getElementById('record-btn');
const recordIcon = document.getElementById('record-icon');
const recordStatus = document.getElementById('record-status');
const timerDisplay = document.getElementById('timer');
const sourceTextDiv = document.getElementById('source-text');
const translatedTextDiv = document.getElementById('translated-text');
const audioResult = document.getElementById('audio-result');
const audioPlayer = document.getElementById('audio-player');
const loadingOverlay = document.getElementById('loading-overlay');

recordBtn.addEventListener('click', toggleRecording);

async function toggleRecording() {
    if (!isRecording) {
        startRecording();
    } else {
        stopRecording();
    }
}

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = sendAudioToBackend;

        mediaRecorder.start();
        isRecording = true;
        
        // UI Updates
        recordBtn.classList.add('recording');
        recordIcon.classList.remove('fa-microphone');
        recordIcon.classList.add('fa-stop');
        recordStatus.innerText = 'Recording...';
        timerDisplay.classList.remove('hidden');
        
        startTime = Date.now();
        updateTimer();
        timerInterval = setInterval(updateTimer, 1000);

    } catch (err) {
        console.error("Error accessing microphone:", err);
        alert("Could not access microphone. Please check permissions.");
    }
}

function stopRecording() {
    mediaRecorder.stop();
    mediaRecorder.stream.getTracks().forEach(track => track.stop());
    isRecording = false;

    // UI Updates
    recordBtn.classList.remove('recording');
    recordIcon.classList.remove('fa-stop');
    recordIcon.classList.add('fa-microphone');
    recordStatus.innerText = 'Click to Record';
    timerDisplay.classList.add('hidden');
    clearInterval(timerInterval);
}

function updateTimer() {
    const elapsed = Math.floor((Date.now() - startTime) / 1000);
    const mins = Math.floor(elapsed / 60).toString().padStart(2, '0');
    const secs = (elapsed % 60).toString().padStart(2, '0');
    timerDisplay.innerText = `${mins}:${secs}`;
}

async function sendAudioToBackend() {
    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.webm');
    formData.append('target_lang', document.getElementById('target-lang').value);
    formData.append('preserve_voice', document.getElementById('preserve-voice').checked);
    formData.append('vc_strength', document.getElementById('vc-strength').value);

    // Show loading
    loadingOverlay.classList.remove('hidden');

    try {
        const response = await fetch('/api/translate', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.error) {
            alert("Error: " + result.error);
        } else {
            sourceTextDiv.innerText = result.source_text;
            translatedTextDiv.innerText = result.translated_text;
            
            if (result.audio_url) {
                audioPlayer.src = result.audio_url + '?t=' + Date.now(); // Cache busting
                audioResult.classList.remove('hidden');
                audioPlayer.play();
            }
        }
    } catch (err) {
        console.error("Upload failed:", err);
        alert("Server communication failed.");
    } finally {
        loadingOverlay.classList.add('hidden');
    }
}
