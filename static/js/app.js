(function () {
    "use strict";

    // --- DOM refs ---
    const screens = {
        welcome: document.getElementById("screen-welcome"),
        capture: document.getElementById("screen-capture"),
        upload: document.getElementById("screen-upload"),
        processing: document.getElementById("screen-processing"),
        result: document.getElementById("screen-result"),
    };

    const webcamVideo = document.getElementById("webcam");
    const overlayCanvas = document.getElementById("overlay-canvas");
    const resultImage = document.getElementById("result-image");
    const uploadPreview = document.getElementById("upload-preview");
    const dropZone = document.getElementById("drop-zone");
    const dropContent = dropZone.querySelector(".drop-zone-content");
    const dropPreview = dropZone.querySelector(".drop-zone-preview");
    const fileInput = document.getElementById("file-input");
    const errorToast = document.getElementById("error-toast");
    const errorMessage = document.getElementById("error-message");

    const btnCamera = document.getElementById("btn-camera");
    const btnUpload = document.getElementById("btn-upload");
    const btnSnap = document.getElementById("btn-snap");
    const btnBackCapture = document.getElementById("btn-back-capture");
    const btnPickFile = document.getElementById("btn-pick-file");
    const btnProcessUpload = document.getElementById("btn-process-upload");
    const btnBackUpload = document.getElementById("btn-back-upload");
    const btnDownload = document.getElementById("btn-download");
    const btnStartOver = document.getElementById("btn-start-over");
    const btnDismissError = document.getElementById("btn-dismiss-error");

    const swatches = document.querySelectorAll(".swatch");

    let webcamStream = null;
    let overlayRAF = null;
    let uploadedFile = null;
    let stepInterval = null;
    let baseImage = null; // stores the transparent PNG as an Image element

    // --- Screen transitions ---
    function showScreen(name) {
        const current = document.querySelector(".screen.active");
        const next = screens[name];
        if (current === next) return;

        if (current) {
            current.classList.add("fade-out");
            setTimeout(() => {
                current.classList.remove("active", "fade-out");
                next.classList.add("active");
            }, 200);
        } else {
            next.classList.add("active");
        }
    }

    // --- Error handling ---
    function showError(msg) {
        errorMessage.textContent = msg;
        errorToast.hidden = false;
        setTimeout(() => { errorToast.hidden = true; }, 6000);
    }

    btnDismissError.addEventListener("click", () => { errorToast.hidden = true; });

    // --- Webcam ---
    async function startWebcam() {
        try {
            webcamStream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 960 } },
                audio: false,
            });
            webcamVideo.srcObject = webcamStream;
            await webcamVideo.play();
            startOverlay();
        } catch (err) {
            showError("Could not access camera. Please allow camera permissions.");
            showScreen("welcome");
        }
    }

    function stopWebcam() {
        if (overlayRAF) {
            cancelAnimationFrame(overlayRAF);
            overlayRAF = null;
        }
        if (webcamStream) {
            webcamStream.getTracks().forEach(t => t.stop());
            webcamStream = null;
        }
        webcamVideo.srcObject = null;
    }

    // --- Silhouette overlay ---
    function startOverlay() {
        const canvas = overlayCanvas;

        function draw() {
            const w = canvas.clientWidth;
            const h = canvas.clientHeight;
            if (canvas.width !== w * 2 || canvas.height !== h * 2) {
                canvas.width = w * 2;
                canvas.height = h * 2;
            }
            const ctx = canvas.getContext("2d");
            const cw = canvas.width;
            const ch = canvas.height;
            ctx.clearRect(0, 0, cw, ch);

            // Semi-transparent dark overlay
            ctx.fillStyle = "rgba(0, 0, 0, 0.45)";
            ctx.fillRect(0, 0, cw, ch);

            // Cut out head + shoulders silhouette
            ctx.save();
            ctx.globalCompositeOperation = "destination-out";
            ctx.fillStyle = "white";
            ctx.beginPath();

            const cx = cw / 2;
            const headRx = cw * 0.15;
            const headRy = cw * 0.19;
            const headCy = ch * 0.34;

            // Head oval
            ctx.ellipse(cx, headCy, headRx, headRy, 0, 0, Math.PI * 2);
            ctx.fill();

            // Neck
            const neckW = headRx * 0.55;
            const neckTop = headCy + headRy * 0.85;
            const neckBottom = headCy + headRy * 1.4;
            ctx.beginPath();
            ctx.rect(cx - neckW, neckTop, neckW * 2, neckBottom - neckTop);
            ctx.fill();

            // Shoulders arc
            ctx.beginPath();
            ctx.ellipse(cx, neckBottom + cw * 0.02, cw * 0.38, cw * 0.22, 0, Math.PI, 0, true);
            ctx.fill();

            ctx.restore();

            // Dashed guide outline
            ctx.save();
            ctx.setLineDash([8, 6]);
            ctx.strokeStyle = "rgba(255, 255, 255, 0.6)";
            ctx.lineWidth = 2;

            // Head outline
            ctx.beginPath();
            ctx.ellipse(cx, headCy, headRx, headRy, 0, 0, Math.PI * 2);
            ctx.stroke();

            // Shoulder arc outline
            ctx.beginPath();
            ctx.ellipse(cx, neckBottom + cw * 0.02, cw * 0.38, cw * 0.22, 0, Math.PI, 0, true);
            ctx.stroke();

            ctx.restore();

            overlayRAF = requestAnimationFrame(draw);
        }

        draw();
    }

    // --- Capture photo from webcam ---
    function capturePhoto() {
        const vw = webcamVideo.videoWidth;
        const vh = webcamVideo.videoHeight;
        const tempCanvas = document.createElement("canvas");
        tempCanvas.width = vw;
        tempCanvas.height = vh;
        const ctx = tempCanvas.getContext("2d");

        // Un-mirror: flip horizontally
        ctx.translate(vw, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(webcamVideo, 0, 0, vw, vh);

        stopWebcam();

        return new Promise(resolve => {
            tempCanvas.toBlob(resolve, "image/jpeg", 0.92);
        });
    }

    // --- Processing steps animation ---
    function startStepAnimation() {
        const steps = document.querySelectorAll(".step");
        steps.forEach(s => { s.classList.remove("active", "done"); });
        let current = 0;
        steps[0].classList.add("active");

        stepInterval = setInterval(() => {
            if (current < steps.length) {
                steps[current].classList.remove("active");
                steps[current].classList.add("done");
            }
            current++;
            if (current < steps.length) {
                steps[current].classList.add("active");
            }
        }, 1200);
    }

    function stopStepAnimation() {
        if (stepInterval) {
            clearInterval(stepInterval);
            stepInterval = null;
        }
        // Mark all done
        document.querySelectorAll(".step").forEach(s => {
            s.classList.remove("active");
            s.classList.add("done");
        });
    }

    // --- API call ---
    async function processImage(blob) {
        showScreen("processing");
        startStepAnimation();

        const formData = new FormData();
        formData.append("image", blob, "photo.jpg");

        try {
            const resp = await fetch("/api/process", { method: "POST", body: formData });
            const data = await resp.json();

            if (!resp.ok) {
                throw new Error(data.error || "Processing failed.");
            }

            stopStepAnimation();

            // Load transparent PNG as base image for color compositing
            baseImage = new Image();
            baseImage.onload = () => {
                // Reset swatches to transparent
                swatches.forEach(s => s.classList.remove("active"));
                document.querySelector(".swatch-transparent").classList.add("active");
                applyBackground("");
                setTimeout(() => showScreen("result"), 600);
            };
            baseImage.src = "data:image/png;base64," + data.image;
        } catch (err) {
            stopStepAnimation();
            showError(err.message);
            showScreen("welcome");
        }
    }

    // --- Background color compositing ---
    function applyBackground(color) {
        if (!baseImage) return;
        const size = baseImage.width;
        const canvas = document.createElement("canvas");
        canvas.width = size;
        canvas.height = size;
        const ctx = canvas.getContext("2d");

        if (color) {
            // Draw colored circle behind the image
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(size / 2, size / 2, size / 2, 0, Math.PI * 2);
            ctx.fill();
        }

        ctx.drawImage(baseImage, 0, 0);

        // Update preview
        resultImage.src = canvas.toDataURL("image/png");

        // Update download link
        if (btnDownload.href.startsWith("blob:")) {
            URL.revokeObjectURL(btnDownload.href);
        }
        canvas.toBlob(blob => {
            btnDownload.href = URL.createObjectURL(blob);
        }, "image/png");
    }

    // --- Reset ---
    function resetAll() {
        stopWebcam();
        uploadedFile = null;
        baseImage = null;
        btnProcessUpload.disabled = true;
        dropContent.hidden = false;
        dropPreview.hidden = true;
        uploadPreview.src = "";
        resultImage.src = "";
        if (btnDownload.href.startsWith("blob:")) {
            URL.revokeObjectURL(btnDownload.href);
        }
        btnDownload.href = "#";
        fileInput.value = "";
        document.querySelectorAll(".step").forEach(s => {
            s.classList.remove("active", "done");
        });
        showScreen("welcome");
    }

    // --- Upload handling ---
    function handleFile(file) {
        if (!file || !file.type.startsWith("image/")) {
            showError("Please select an image file.");
            return;
        }
        uploadedFile = file;
        const url = URL.createObjectURL(file);
        uploadPreview.src = url;
        dropContent.hidden = true;
        dropPreview.hidden = false;
        btnProcessUpload.disabled = false;
    }

    // --- Event listeners ---

    // Welcome screen
    btnCamera.addEventListener("click", () => {
        showScreen("capture");
        startWebcam();
    });

    btnUpload.addEventListener("click", () => {
        showScreen("upload");
    });

    // Capture screen
    btnSnap.addEventListener("click", async () => {
        const blob = await capturePhoto();
        if (blob) processImage(blob);
    });

    btnBackCapture.addEventListener("click", () => {
        stopWebcam();
        showScreen("welcome");
    });

    // Upload screen
    btnPickFile.addEventListener("click", () => fileInput.click());

    fileInput.addEventListener("change", () => {
        if (fileInput.files.length > 0) handleFile(fileInput.files[0]);
    });

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("drag-active");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("drag-active");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("drag-active");
        if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
    });

    btnProcessUpload.addEventListener("click", () => {
        if (uploadedFile) processImage(uploadedFile);
    });

    btnBackUpload.addEventListener("click", () => {
        uploadedFile = null;
        btnProcessUpload.disabled = true;
        dropContent.hidden = false;
        dropPreview.hidden = true;
        uploadPreview.src = "";
        fileInput.value = "";
        showScreen("welcome");
    });

    // Result screen
    swatches.forEach(swatch => {
        swatch.addEventListener("click", () => {
            swatches.forEach(s => s.classList.remove("active"));
            swatch.classList.add("active");
            applyBackground(swatch.dataset.color);
        });
    });

    btnStartOver.addEventListener("click", resetAll);
})();
