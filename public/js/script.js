const CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

let correctLabel;
let guess;

// linNetActs become flattened when read, so we have to be careful with indexing
let linNetActs; 
let linPredictions;
let mlpNetActs;
let mlpPredictions;
let leakyNetActs;
let leakyPredictions;

let testBatchBuffer; // Buffer for reading in binary data

let count; // Index of the image

let correctGuesses = 0;
let totalGuesses = 0;

let state = 'GUESSING';

const canvas = document.getElementById("fireworksCanvas");
const ctx = canvas.getContext("2d");

canvas.width = window.innerWidth;
canvas.height = window.innerHeight;
window.addEventListener("resize", () => [canvas.width, canvas.height] = [window.innerWidth, window.innerHeight], false);

class Firework {
    constructor() {
        this.x = Math.random() * canvas.width;
        this.y = canvas.height;
        this.sx = Math.random() * 3 - 1.5;
        this.sy = Math.random() * -3;
        this.size = Math.random() * 2 + 1;
        const colorVal = Math.floor(Math.random() * 0xffffff);
        this.r = (colorVal >> 16) & 255;
        this.g = (colorVal >> 8) & 255;
        this.b = colorVal & 255;
        this.shouldExplode = false;
    }

    update() {
        this.x += this.sx;
        this.y += this.sy;
        this.sy += 0.05;
        this.shouldExplode = this.sy >= 2 || this.y <= 100 || this.x <= 0 || this.x >= canvas.width;
    }

    draw() {
        ctx.fillStyle = `rgb(${this.r},${this.g},${this.b})`;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        ctx.fill();
    }
}

class Particle {
    constructor(x, y, r, g, b) {
        this.x = x;
        this.y = y;
        this.sx = Math.random() * 3 - 1.5;
        this.sy = Math.random() * 3 - 1.5;
        this.r = r;
        this.g = g;
        this.b = b;
        this.size = Math.random() * 2 + 1;
        this.life = 100;
    }
    update() {
        this.x += this.sx;
        this.y += this.sy;
        this.life -= 1;
    }
    draw() {
        ctx.fillStyle = `rgba(${this.r},${this.g},${this.b},${this.life / 100})`;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        ctx.fill();
    }
}

const fireworks = [];
const particles = [];
let fireworksActive = false;

function animate() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (fireworksActive && Math.random() < 0.25) fireworks.push(new Firework());

    fireworks.forEach((firework, i) => {
        firework.update();
        firework.draw();
        if (firework.shouldExplode) {
            for (let j = 0; j < 50; j++) {
                particles.push(new Particle(firework.x, firework.y, firework.r, firework.g, firework.b));
            }
            fireworks.splice(i, 1);
        }
    });

    particles.forEach((particle, i) => {
        particle.update();
        particle.draw();
        if (particle.life <= 0) particles.splice(i, 1);
    });

    requestAnimationFrame(animate);
}


animate();

function displayState(curState) {
    state = curState;
    switch (state) {
        case 'GUESSING':
            document.getElementById('label').style.display = "none";

            document.getElementById('softmax-prediction').style.display = "none";
            document.getElementById('mlp-prediction').style.display = "none";
            document.getElementById('leaky-prediction').style.display = "none";

            document.getElementById('guess-status').innerText = "";
            break;
        case 'REVEALING':
            document.getElementById('label').style.display = "block";

            document.getElementById('softmax-prediction').style.display = "block";
            document.getElementById('mlp-prediction').style.display = "block";
            document.getElementById('leaky-prediction').style.display = "block";
            
            guess_element = document.getElementById('guess-status');
            if (guess.toLowerCase() === correctLabel) {
                guess_element.innerText = "Correct!";
                guess_element.style.color = "green";
                fireworksActive = true;
                setTimeout(() => { fireworksActive = false; }, 4000);
                correctGuesses++;
            } else {
                guess_element.innerText = "Incorrect.";
                guess_element.style.color = "red";
            }
            break;
    }
}

function displayImage(arrayBuffer, imageIndex, canvasId) {
    const RECORD_SIZE = 3073; // label + 3072 byes
    const offset = imageIndex * RECORD_SIZE;

    const record = new Uint8Array(arrayBuffer, offset, RECORD_SIZE);
    const labelIdx = record[0];
    correctLabel = CLASSES[labelIdx];

    // See cifar-10 website for more info; every 1024 values is a different color
    const rDimension = record.slice(1, 1 + 1024);
    const gDimension = record.slice(1 + 1024, 1 + 1024 * 2);
    const bDimension = record.slice(1 + 1024 * 2, 1 + 1024 * 3);

    const canvas = document.getElementById(canvasId);
    const ctx = canvas.getContext('2d');
    const imageData = ctx.createImageData(32, 32);

    for (let i = 0; i < 1024; i++) {
        const pixelIdx = i * 4;
        imageData.data[pixelIdx] = rDimension[i];
        imageData.data[pixelIdx + 1] = gDimension[i];
        imageData.data[pixelIdx + 2] = bDimension[i];
        imageData.data[pixelIdx + 3] = 255; // Full opacity
    }

    document.getElementById('loading-canvas').style.display = "none";
    ctx.putImageData(imageData, 0, 0);
    document.getElementById('label').innerText = correctLabel;
}

async function getDataset() {
    try {
        const [linNetActsBin, linPredictionsBin, 
            mlpNetActsBin, mlpPredictionsBin, 
            leakyNetActsBin, leakyPredictionsBin,
            testBatchBin] = await Promise.all([
            fetch('./cifar-10-binary/cifar-10-batches-bin/linear/lin_net_acts.bin'),
            fetch('./cifar-10-binary/cifar-10-batches-bin/linear/lin_preds.bin'),

            fetch('./cifar-10-binary/cifar-10-batches-bin/mlp/mlp_net_acts.bin'),
            fetch('./cifar-10-binary/cifar-10-batches-bin/mlp/mlp_preds.bin'),

            fetch('./cifar-10-binary/cifar-10-batches-bin/leaky/leaky_net_acts.bin'),
            fetch('./cifar-10-binary/cifar-10-batches-bin/leaky/leaky_preds.bin'), 

            fetch('./cifar-10-binary/cifar-10-batches-bin/test_batch.bin')
        ]);

        linNetActs = new Float32Array(await linNetActsBin.arrayBuffer());
        linPredictions = new Uint32Array(await linPredictionsBin.arrayBuffer());

        mlpNetActs = new Float32Array(await mlpNetActsBin.arrayBuffer());
        mlpPredictions = new Uint32Array(await mlpPredictionsBin.arrayBuffer());

        leakyNetActs = new Float32Array(await leakyNetActsBin.arrayBuffer());
        leakyPredictions = new Uint32Array(await leakyPredictionsBin.arrayBuffer());
        
        testBatchBuffer = await testBatchBin.arrayBuffer();

        getNextImage(true);
    } catch (error) {
        console.error("Error loading dataset:", error);
    }
}

function getNextImage(isFirstLoad=false) {
    if (!isFirstLoad && (!testBatchBuffer || state === 'GUESSING')) {
        return;
    }

    displayState('GUESSING');

    count = Math.floor(10_000 * Math.random());
    displayImage(testBatchBuffer, count, 'image-canvas');

    let linPrediction = linPredictions[count];
    let linPredictionLabel = CLASSES[linPrediction];
    let linConfidence = linNetActs[CLASSES.length * count + linPrediction];
    let formattedLinConfidence = (100 * linConfidence).toFixed(2);

    let mlpPrediction = mlpPredictions[count];
    let mlpPredictionLabel = CLASSES[mlpPrediction];
    let mlpConfidence = mlpNetActs[CLASSES.length * count + mlpPrediction];
    let formattedMlpConfidence = (100 * mlpConfidence).toFixed(2);

    let leakyPrediction = leakyPredictions[count];
    let leakyPredictionLabel = CLASSES[leakyPrediction];
    let leakyConfidence = leakyNetActs[CLASSES.length * count + leakyPrediction];
    let formattedLeakyConfidence = (100 * leakyConfidence).toFixed(2);

    let linGuessStatus = linPredictionLabel === correctLabel ? "correctly" : "incorrectly";
    document.getElementById("softmax-prediction").innerText = 
        `Softmax ${linGuessStatus} predicted ${linPredictionLabel} with a confidence of ${formattedLinConfidence}%`;

    let mlpGuessStatus = mlpPredictionLabel === correctLabel ? "correctly" : "incorrectly";
    document.getElementById("mlp-prediction").innerText = 
        `\nMLP ${mlpGuessStatus} predicted ${mlpPredictionLabel} with a confidence of ${formattedMlpConfidence}%`;

    let leakyGuessStatus = leakyPredictionLabel === correctLabel ? "correctly" : "incorrectly";
    document.getElementById("leaky-prediction").innerText = 
        `\nMLP with Leaky-ReLU ${leakyGuessStatus} predicted ${leakyPredictionLabel} with a confidence of ${formattedLeakyConfidence}%`;
    
}

function submitGuess() {
    guess = document.getElementById("guess").value;
    if (state === 'REVEALING') {
        if (!guess) { // User can hit enter quickly after guessing to move to next image
            getNextImage();
        } else { // Prevents user from submitting a guess multiple times for same image
            return;
        }
    }

    if (!guess) {
        return;
    }

    displayState('REVEALING');

    totalGuesses++;
    document.getElementById("guess").value = ""; // Clear guess box
    document.getElementById("score").innerText = `Score: ${correctGuesses}/${totalGuesses}`;
}

function addDatalistOptions() {
    const datalist = document.getElementById("cifar-classes");
    datalist.innerHTML = CLASSES.map(item =>
        `<option value="${item}"></option>`
    ).join('');
}

async function startApp() {
    await getDataset();
    addDatalistOptions();
}

startApp();
