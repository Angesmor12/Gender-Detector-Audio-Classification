const recordingText = document.querySelector(".recording-text-container")
const predictionText = document.querySelector(".prediction-value")
const predictionContainer = document.querySelector(".prediction-container")
const canvas = document.querySelector("#spectrogramCanvas");
const ctx = canvas.getContext('2d');

let allow = 1
let loadingImage = document.querySelector(".loading-image-container")
let path = "./models/model_8_cnn.onnx"
let targetSampleRate = 8000
let sampleRateLength = 24000
let fftSize = 726
let loadPath = ""
let session = ""

document.querySelector(".algorithm-input").addEventListener("change", (e)=>{
  path = e.target.value
  targetSampleRate = parseInt(e.target.selectedOptions[0].dataset.sample_rate)
  sampleRateLength = parseInt(e.target.selectedOptions[0].dataset.sample_rate_length)
  fftSize = parseInt(e.target.selectedOptions[0].dataset.fft_size)
})

async function predict(inputFeatures,path, key) {

    if (loadPath != path)
    {  
      session = await ort.InferenceSession.create(path);
    }

    const flattenedFeatures = inputFeatures.flatMap(arr => Array.from(arr));

    const tensor = new ort.Tensor('float32', flattenedFeatures, [1, 1, 128, 121]);

    const feeds = {};
    feeds[key] = tensor

    const result = await session.run(feeds);

    const logits = result.output.data; 
    const probabilities = softmax(logits);

    loadPath = path

    return probabilities;
}

const uploadArea = document.querySelector('.upload-area');

uploadArea.addEventListener('dragover', (event) => {
    event.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (event) => {
    event.preventDefault();
    uploadArea.classList.remove('dragover');

    const files = event.dataTransfer.files;
    if (files.length) {
        handleFiles(files);
    }
});

uploadArea.addEventListener('click', () => {
  const input = document.createElement('input');
  input.type = 'file';
  input.multiple = true; 
  input.click();

  input.addEventListener('change', (event) => {
      const files = event.target.files;
      if (files.length) {
          handleFiles(files);
      }
  });
});

function softmax(logits) {
  const maxLogit = Math.max(...logits); 
  const expLogits = logits.map(logit => Math.exp(logit - maxLogit));
  const sumExp = expLogits.reduce((a, b) => a + b, 0);
  return expLogits.map(value => value / sumExp);
}

// Process audio 

async function playAudio(waveform, sampleRate) {

  const audioContext = new (window.AudioContext || window.webkitAudioContext)();

  const audioBuffer = audioContext.createBuffer(1, waveform.length, sampleRate); 
  audioBuffer.copyToChannel(waveform, 0, 0); 

  const source = audioContext.createBufferSource();
  source.buffer = audioBuffer;

  source.connect(audioContext.destination);

  source.start();

  return new Promise((resolve) => {
    source.onended = () => {
      resolve();
    };
  });
}

function convertStereoToMono(audioBuffer) {
  const channel1 = audioBuffer.getChannelData(0); 
  const channel2 = audioBuffer.getChannelData(1); 
  const monoWaveform = new Float32Array(channel1.length);

  for (let i = 0; i < channel1.length; i++) {
    monoWaveform[i] = (channel1[i] + channel2[i]) / 2; 
  }

  return monoWaveform;
}

async function handleFiles(files) {
  for (const file of files) {

    if (
      file.type !== "audio/mpeg" &&
      file.type !== "audio/wav" &&
      file.type !== "audio/ogg" &&
      file.type !== "audio/aac" &&
      file.type !== "audio/flac" &&
      file.type !== "audio/webm" &&
      file.type !== "audio/x-m4a" &&
      file.type !== "audio/m4a"
    ) {
      loadingImage.classList.add("hidden");
      return window.alert(
        "The file must be an audio file (MP3, WAV, OGG, AAC, FLAC, WebM)."
      );
    }

    processAudio(file, false, false);

  }
}

async function processAudio(file, normalization, draw) {

  loadingImage.classList.remove("hidden");
  predictionContainer.classList.add("invisible");

  const arrayBuffer = await file.arrayBuffer();

  const audioContext = new (window.AudioContext || window.webkitAudioContext)();

  const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

  let waveform = audioBuffer.getChannelData(0);

  if (audioBuffer.numberOfChannels > 1) {
    waveform = convertStereoToMono(audioBuffer);
  }

  let sampleRate = audioBuffer.sampleRate;

  if (sampleRate !== targetSampleRate) {

    const resampleFactor = targetSampleRate / sampleRate;
    const resampledWaveform = new Float32Array(Math.floor(waveform.length * resampleFactor));

    for (let i = 0; i < resampledWaveform.length; i++) {
      const originalIndex = i / resampleFactor;
      const lowerIndex = Math.floor(originalIndex);
      const upperIndex = Math.min(lowerIndex + 1, waveform.length - 1);
      const interpolation = originalIndex - lowerIndex;
      resampledWaveform[i] = waveform[lowerIndex] * (1 - interpolation) + waveform[upperIndex] * interpolation;
    }

    waveform = resampledWaveform;
    sampleRate = targetSampleRate;

  }

  if (waveform.length > sampleRateLength) {
    waveform = waveform.slice(0, sampleRateLength);
  } else if (waveform.length < sampleRateLength) {
    const paddedWaveform = new Float32Array(sampleRateLength);
    paddedWaveform.set(waveform, 0);
    waveform = paddedWaveform;
  }

  if (normalization){
    waveform = normalizeWaveform(waveform);
  }

  const spectrogram = generateMelSpectrogram(waveform, targetSampleRate, canvas, fftSize, draw);

  const prediction = await predict(spectrogram,path, "input")

  showPrediction(prediction)
}

function normalizeWaveform(waveform) {
  const max = Math.max(...waveform);
  const min = Math.min(...waveform);
  const absMax = Math.max(Math.abs(max), Math.abs(min));

  if (absMax > 0) {
    return waveform.map(value => value / absMax);
  }

  return waveform;
}

function showPrediction(value){

  const classValue = value.indexOf(Math.max(...value))
  loadingImage.classList.add("hidden")
  predictionContainer.classList.remove("invisible")

  if (classValue == 0){
      predictionText.innerHTML = "Male"
  }
  else {
      predictionText.innerHTML = "Female"
  }
}

let attempt = 1

const modalBody = document.querySelector(".md")
const modalBody2 = document.querySelector(".md2")

document.querySelector(".option-b1 button").addEventListener("click", ()=>{
  allow = 0
  modalBody.classList.remove("hidden")
  modalBody2.classList.add("hidden")
})

document.querySelector(".option-b2 button").addEventListener("click", async ()=>{

  let result = false

  if (!setupRecorderSesion)
  {  
    result = await setupRecorder();
    setupRecorderSesion = result
  }

  modalBody.classList.add("hidden")
  modalBody2.classList.remove("hidden")
  allow = 1
})

// Use microphone

const startButton = document.getElementById('start-btn');
const audioPlayer = document.getElementById('audio-player');
let setupRecorderSesion = false;

let mediaRecorder;
let audioChunks = []; 

async function setupRecorder() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.ondataavailable = (event) => {
      audioChunks.push(event.data);
    };
    
    mediaRecorder.onstop = () => {
      
      const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
      const audioURL = URL.createObjectURL(audioBlob);
      audioPlayer.src = audioURL;
      audioChunks = [];

      processAudio(audioBlob, false, true);

    };

    startButton.disabled = false;

    return true
  } catch (error) {
    startButton.disabled = true;
    alert('The microphone could not be accessed.');
    return false
  }
}

// Start recording
startButton.addEventListener('click', () => {
  if (mediaRecorder) {

    recordingText.classList.remove("hidden")
    predictionContainer.classList.add("invisible")

    mediaRecorder.start();

    startButton.disabled = true;

    setTimeout(() => {
      mediaRecorder.stop();
      startButton.disabled = false;
      recordingText.classList.add("hidden")
    }, 3500); 

  }
});

// Generate spectrogram-Mel

function generateMelSpectrogram(waveform, sampleRate, canvas, fftSize, draw) {

  const hopSize = fftSize / 4;
  const numMelBands = 121; 
  const spectrogram = [];
  const numBins = fftSize / 2;
  const numFrames = Math.floor((waveform.length - fftSize) / hopSize);

  const fft = new Float32Array(fftSize);
  const windowFunction = new Float32Array(fftSize);

  for (let i = 0; i < fftSize; i++) {
    windowFunction[i] = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (fftSize - 1));
  }

  const melFilterBank = createMelFilterBank(numMelBands, numBins, sampleRate, fftSize);

  for (let frame = 0; frame < numFrames; frame++) {
    const start = frame * hopSize;
    const slice = waveform.subarray(start, start + fftSize);

    for (let i = 0; i < fftSize; i++) {
      fft[i] = slice[i] * windowFunction[i];
    }

    const spectrum = new Float32Array(numBins);
    for (let k = 0; k < numBins; k++) {
      let sumReal = 0;
      let sumImag = 0;
      for (let n = 0; n < fftSize; n++) {
        const angle = (-2 * Math.PI * k * n) / fftSize;
        sumReal += fft[n] * Math.cos(angle);
        sumImag += fft[n] * Math.sin(angle);
      }
      spectrum[k] = Math.sqrt(sumReal ** 2 + sumImag ** 2);
    }

    const melSpectrum = new Float32Array(numMelBands);
    for (let melBand = 0; melBand < numMelBands; melBand++) {
      melSpectrum[melBand] = 0;
      for (let bin = 0; bin < numBins; bin++) {
        melSpectrum[melBand] += spectrum[bin] * melFilterBank[melBand][bin];
      }
    }

    spectrogram.push(melSpectrum);
  }

  if (draw){
    drawSpectrogram(spectrogram, canvas);
  }

  return spectrogram
}

function createMelFilterBank(numMelBands, numBins, sampleRate) {
  const melFilterBank = [];
  const nyquist = sampleRate / 2;
  const melMin = 0;
  const melMax = hzToMel(nyquist);
  const melPoints = new Float32Array(numMelBands + 2);

  for (let i = 0; i < melPoints.length; i++) {
    melPoints[i] = melMin + (melMax - melMin) * (i / (numMelBands + 1));
  }

  const freqPoints = melPoints.map(melToHz);
  const binFrequencies = Array.from({ length: numBins }, (_, i) => (i * nyquist) / numBins);

  for (let melBand = 0; melBand < numMelBands; melBand++) {
    const filter = new Float32Array(numBins);
    for (let bin = 0; bin < numBins; bin++) {
      const freq = binFrequencies[bin];
      if (freq >= freqPoints[melBand] && freq <= freqPoints[melBand + 1]) {
        filter[bin] = (freq - freqPoints[melBand]) / (freqPoints[melBand + 1] - freqPoints[melBand]);
      } else if (freq > freqPoints[melBand + 1] && freq <= freqPoints[melBand + 2]) {
        filter[bin] = (freqPoints[melBand + 2] - freq) / (freqPoints[melBand + 2] - freqPoints[melBand + 1]);
      }
    }
    melFilterBank.push(filter);
  }

  return melFilterBank;
}

function hzToMel(hz) {
  return 2595 * Math.log10(1 + hz / 700);
}

function melToHz(mel) {
  return 700 * (Math.pow(10, mel / 2595) - 1);
}

function drawSpectrogram(spectrogram, canvas) {
  
  const ctx = canvas.getContext('2d');

  const width = canvas.width;
  const height = canvas.height;

  const numFrames = spectrogram.length;
  const numBands = spectrogram[0].length;
  const imageData = ctx.createImageData(width, height);

  for (let x = 0; x < numFrames; x++) {
    for (let y = 0; y < numBands; y++) {
      const intensity = spectrogram[x][y];
      const colorValue = Math.min(255, Math.floor(Math.log1p(intensity) * 255));
      const index = (x + (numBands - 1 - y) * width) * 4;
      imageData.data[index] = colorValue;
      imageData.data[index + 1] = colorValue;
      imageData.data[index + 2] = colorValue;
      imageData.data[index + 3] = 255; 
    }
  }

  ctx.putImageData(imageData, 0, 0);
}
