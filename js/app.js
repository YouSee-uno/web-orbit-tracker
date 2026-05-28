/**
 * Orbit Tracker - Application Controller & WebRTC Camera Manager
 * Camera starts immediately on load. OpenCV loads asynchronously in background.
 */

document.addEventListener('DOMContentLoaded', () => {
    // === UI Elements ===
    const video             = document.getElementById('camera-video');
    const drawingCanvas     = document.getElementById('drawing-canvas');
    const drawingCtx        = drawingCanvas.getContext('2d');
    const viewportContainer = document.getElementById('viewport-container');
    const loadingOverlay    = document.getElementById('loading-overlay');
    const loadingStatusText = document.getElementById('loading-status');
    const opencvBadge       = document.getElementById('opencv-status-badge');

    // HUD Stats
    const fpsValue          = document.getElementById('fps-value');
    const statusBadge       = document.getElementById('status-badge');
    const trailPointsValue  = document.getElementById('trail-points-value');
    const targetCoordsValue = document.getElementById('target-coords-value');

    // Controls
    const startRecordBtn   = document.getElementById('btn-start-record');
    const stopRecordBtn    = document.getElementById('btn-stop-record');
    const rangeExpandBtn   = document.getElementById('btn-range-expand');
    const rangeShrinkBtn   = document.getElementById('btn-range-shrink');
    const trailResetBtn    = document.getElementById('btn-trail-reset');
    const cameraSelectorBtn = document.getElementById('btn-camera-select');
    const cameraPickerEl   = document.getElementById('camera-picker');
    const recordBtnLabel   = document.getElementById('record-btn-label');

    // === App State ===
    let activeStream    = null;
    let frameId         = null;
    let opencvReady     = false;
    let opencvFailed    = false;
    let isRecording     = false;
    let selectedPoint   = null;
    let cameraDevices   = [];
    let activeCameraId  = null;

    // Performance metrics
    let lastFrameTime = performance.now();
    let fpsHistory    = [];

    // Tracker instance (OpenCV ops guarded by opencvReady flag)
    const tracker = new OrbitTracker();

    // Hidden 640×480 canvas for OpenCV processing
    const processWidth  = 640;
    const processHeight = 480;
    const processCanvas = document.createElement('canvas');
    processCanvas.width  = processWidth;
    processCanvas.height = processHeight;
    const processCtx = processCanvas.getContext('2d', { willReadFrequently: true });

    const activePaletteColor = '#10b981';

    // === Canvas Resize ===
    function resizeCanvas() {
        const rect = viewportContainer.getBoundingClientRect();
        drawingCanvas.width  = rect.width;
        drawingCanvas.height = rect.height;
    }
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();

    // === Drag / Tap State for Region Selection ===
    let dragStart = null;
    let isDragging = false;
    let dragRect = null; // canvas-space { x, y, w, h } while dragging

    // Convert a canvas-space point to process-space, accounting for mirror
    function toProcessCoords(canvasX, canvasY, canvasRect) {
        const scaleX = processWidth  / canvasRect.width;
        const scaleY = processHeight / canvasRect.height;
        let lx = canvasX * scaleX;
        let ly = canvasY * scaleY;
        if (video.style.transform.includes('scaleX(-1)')) lx = processWidth - lx;
        return { x: lx, y: ly };
    }

    // Immediately start tracking at a canvas-space point
    function initiateTrackingAtPoint(canvasX, canvasY) {
        const canvasRect = drawingCanvas.getBoundingClientRect();
        const p = toProcessCoords(canvasX, canvasY, canvasRect);
        selectedPoint = { processX: p.x, processY: p.y, canvasX, canvasY };
        drawTouchRipple(canvasX, canvasY);

        if (opencvReady) {
            processCtx.drawImage(video, 0, 0, processWidth, processHeight);
            const imgData = processCtx.getImageData(0, 0, processWidth, processHeight);
            let srcMat = cv.matFromImageData(imgData);
            tracker.initTracking(srcMat, p.x, p.y);
            tracker.recordingStyle = true;
            srcMat.delete();
            activateTrackingUI();
        } else {
            startRecordBtn.disabled = false;
            showToast(opencvFailed
                ? '⚠ 追尾エンジンの読み込みに失敗しました'
                : '⚙ 追尾エンジン読込中... しばらくお待ちください');
        }
    }

    // Immediately start tracking from a canvas-space rectangle selection
    function initiateTrackingAtRect(rect) {
        const canvasRect = drawingCanvas.getBoundingClientRect();
        const scaleX = processWidth  / canvasRect.width;
        const scaleY = processHeight / canvasRect.height;
        const isMirrored = video.style.transform.includes('scaleX(-1)');

        let rx = rect.x * scaleX;
        let ry = rect.y * scaleY;
        let rw = rect.w * scaleX;
        let rh = rect.h * scaleY;
        if (isMirrored) rx = processWidth - rx - rw;

        const cx = rect.x + rect.w / 2;
        const cy = rect.y + rect.h / 2;
        selectedPoint = { processX: rx + rw / 2, processY: ry + rh / 2, canvasX: cx, canvasY: cy };

        if (opencvReady) {
            processCtx.drawImage(video, 0, 0, processWidth, processHeight);
            const imgData = processCtx.getImageData(0, 0, processWidth, processHeight);
            let srcMat = cv.matFromImageData(imgData);
            tracker.initTrackingFromRect(srcMat, rx, ry, rw, rh);
            tracker.recordingStyle = true;
            srcMat.delete();
            activateTrackingUI();
        } else {
            startRecordBtn.disabled = false;
            showToast(opencvFailed
                ? '⚠ 追尾エンジンの読み込みに失敗しました'
                : '⚙ 追尾エンジン読込中... しばらくお待ちください');
        }
    }

    function activateTrackingUI() {
        isRecording = true;
        startRecordBtn.style.display = 'none';
        stopRecordBtn.style.display  = 'flex';
        recordBtnLabel.textContent   = '撮影終了';
    }

    // === Pointer Events for Tap + Drag-Select ===
    drawingCanvas.addEventListener('pointerdown', (e) => {
        if (!activeStream) return;
        e.preventDefault();
        const rect = drawingCanvas.getBoundingClientRect();
        dragStart = { x: e.clientX - rect.left, y: e.clientY - rect.top };
        isDragging = false;
        dragRect = null;
        drawingCanvas.setPointerCapture(e.pointerId);
    });

    drawingCanvas.addEventListener('pointermove', (e) => {
        if (!dragStart) return;
        const rect = drawingCanvas.getBoundingClientRect();
        const cx = e.clientX - rect.left;
        const cy = e.clientY - rect.top;
        const dx = cx - dragStart.x;
        const dy = cy - dragStart.y;
        if (Math.abs(dx) > 12 || Math.abs(dy) > 12) {
            isDragging = true;
            dragRect = {
                x: Math.min(dragStart.x, cx),
                y: Math.min(dragStart.y, cy),
                w: Math.abs(dx),
                h: Math.abs(dy)
            };
        }
    });

    drawingCanvas.addEventListener('pointerup', () => {
        if (!dragStart) return;

        if (isDragging && dragRect && dragRect.w > 15 && dragRect.h > 15) {
            initiateTrackingAtRect(dragRect);
        } else {
            initiateTrackingAtPoint(dragStart.x, dragStart.y);
        }

        dragStart  = null;
        isDragging = false;
        dragRect   = null;
    });

    // === Ripple Feedback ===
    function drawTouchRipple(x, y) {
        const startTime = performance.now();
        const duration  = 400;
        function animateRipple() {
            const progress = (performance.now() - startTime) / duration;
            if (progress < 1.0) {
                window.touchRipple = { x, y, progress };
                requestAnimationFrame(animateRipple);
            } else {
                window.touchRipple = null;
            }
        }
        animateRipple();
    }

    // === Toast notification for user feedback ===
    function showToast(msg) {
        let toast = document.getElementById('app-toast');
        if (!toast) {
            toast = document.createElement('div');
            toast.id = 'app-toast';
            Object.assign(toast.style, {
                position: 'absolute', bottom: '80px', left: '50%',
                transform: 'translateX(-50%)',
                background: 'rgba(13,20,35,0.85)', border: '1px solid rgba(255,255,255,0.1)',
                backdropFilter: 'blur(12px)', color: '#f8fafc',
                fontFamily: "'Outfit', sans-serif", fontSize: '0.9rem', fontWeight: '500',
                padding: '10px 22px', borderRadius: '12px', zIndex: '50',
                transition: 'opacity 0.4s ease'
            });
            document.getElementById('app-container').appendChild(toast);
        }
        toast.textContent = msg;
        toast.style.opacity = '1';
        clearTimeout(toast._timer);
        toast._timer = setTimeout(() => { toast.style.opacity = '0'; }, 2500);
    }

    // === Camera Start ===
    async function startCamera(deviceId = null) {
        if (activeStream) activeStream.getTracks().forEach(t => t.stop());

        const constraints = {
            video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 60 } },
            audio: false
        };
        if (deviceId) {
            constraints.video.deviceId = { exact: deviceId };
        } else {
            constraints.video.facingMode = { ideal: 'environment' };
        }

        try {
            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            activeStream = stream;
            video.srcObject = stream;

            await new Promise(resolve => { video.onloadedmetadata = resolve; });

            const track    = stream.getVideoTracks()[0];
            const settings = track.getSettings();
            activeCameraId = settings.deviceId;
            video.style.transform =
                (settings.facingMode === 'user' || (track.label && track.label.toLowerCase().includes('front')))
                ? 'scaleX(-1)' : 'scaleX(1)';

            // Hide loading overlay — camera is live
            loadingOverlay.style.opacity    = '0';
            loadingOverlay.style.visibility = 'hidden';

            if (!frameId) {
                lastFrameTime = performance.now();
                runProcessingLoop();
            }
        } catch (err) {
            console.error('Camera failed:', err);
            loadingStatusText.innerHTML =
                `<span style="color:#ec4899;font-weight:bold;">カメラのアクセスが拒否されました</span><br>ブラウザのカメラ権限を許可してください。`;
        }
    }

    // === Main rAF Loop ===
    function runProcessingLoop() {
        const now   = performance.now();
        const delta = now - lastFrameTime;
        lastFrameTime = now;
        const fps = Math.round(1000 / delta);
        fpsHistory.push(fps);
        if (fpsHistory.length > 30) fpsHistory.shift();
        fpsValue.textContent = Math.round(fpsHistory.reduce((a, b) => a + b, 0) / fpsHistory.length);

        // OpenCV tracking (only if WASM ready)
        if (opencvReady && tracker.isTracking) {
            let srcMat = null;
            try {
                processCtx.drawImage(video, 0, 0, processWidth, processHeight);
                const imgData = processCtx.getImageData(0, 0, processWidth, processHeight);
                srcMat = cv.matFromImageData(imgData);
                tracker.processFrame(srcMat);
            } catch (err) {
                console.warn('Tracking frame error:', err);
            } finally {
                if (srcMat) srcMat.delete();
            }
        }

        // Clear canvas
        drawingCtx.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);

        const isMirrored = video.style.transform.includes('scaleX(-1)');
        if (isMirrored) {
            drawingCtx.save();
            drawingCtx.translate(drawingCanvas.width, 0);
            drawingCtx.scale(-1, 1);
        }

        tracker.draw(drawingCtx, drawingCanvas.width, drawingCanvas.height);

        if (isMirrored) drawingCtx.restore();

        // Ripple
        if (window.touchRipple) {
            const r = window.touchRipple;
            drawingCtx.save();
            drawingCtx.beginPath();
            drawingCtx.arc(r.x, r.y, r.progress * 45, 0, Math.PI * 2);
            drawingCtx.strokeStyle = activePaletteColor;
            drawingCtx.lineWidth   = 3 * (1.0 - r.progress);
            drawingCtx.globalAlpha = 0.8 * (1.0 - r.progress);
            drawingCtx.stroke();
            drawingCtx.restore();
        }

        // Drag selection rectangle (live preview while dragging)
        if (isDragging && dragRect) {
            drawingCtx.save();
            drawingCtx.strokeStyle = activePaletteColor;
            drawingCtx.lineWidth   = 2;
            drawingCtx.setLineDash([6, 3]);
            drawingCtx.globalAlpha = 0.85;
            drawingCtx.strokeRect(dragRect.x, dragRect.y, dragRect.w, dragRect.h);
            drawingCtx.fillStyle   = activePaletteColor;
            drawingCtx.globalAlpha = 0.1;
            drawingCtx.fillRect(dragRect.x, dragRect.y, dragRect.w, dragRect.h);
            drawingCtx.restore();
        }

        // Selected point marker (shown before recording starts)
        if (selectedPoint && !isRecording) {
            const pulse = 20 + Math.sin(Date.now() / 220) * 3;
            drawingCtx.save();
            drawingCtx.strokeStyle = activePaletteColor;
            drawingCtx.fillStyle   = activePaletteColor;
            drawingCtx.lineWidth   = 2;
            drawingCtx.globalAlpha = 0.85;
            drawingCtx.beginPath();
            drawingCtx.arc(selectedPoint.canvasX, selectedPoint.canvasY, pulse, 0, Math.PI * 2);
            drawingCtx.stroke();
            drawingCtx.beginPath();
            drawingCtx.arc(selectedPoint.canvasX, selectedPoint.canvasY, 3, 0, Math.PI * 2);
            drawingCtx.fill();
            drawingCtx.restore();
        }

        updateStatsHUD();
        frameId = requestAnimationFrame(runProcessingLoop);
    }

    // === HUD Stats ===
    function updateStatsHUD() {
        statusBadge.className = 'status-badge';
        if (!opencvReady) {
            statusBadge.classList.add('status-searching');
            statusBadge.textContent = opencvFailed ? 'ENGINE ERR' : 'LOADING';
        } else if (tracker.status === 'TRACKED') {
            statusBadge.classList.add('status-tracked');
            statusBadge.textContent = 'TRACKED';
        } else if (tracker.status === 'LOST') {
            statusBadge.classList.add('status-lost');
            statusBadge.textContent = 'LOST';
        } else if (tracker.status === 'SEARCHING') {
            statusBadge.classList.add('status-searching');
            statusBadge.textContent = 'SEARCH';
        } else {
            statusBadge.classList.add('status-none');
            statusBadge.textContent = 'STANDBY';
        }

        trailPointsValue.textContent = tracker.history.length;
        if (tracker.isTracking && isRecording) {
            const isMirrored = video.style.transform.includes('scaleX(-1)');
            let displayX = Math.round(tracker.targetX);
            if (isMirrored) displayX = processWidth - displayX;
            targetCoordsValue.textContent = `X: ${displayX}, Y: ${Math.round(tracker.targetY)}`;
        } else if (selectedPoint) {
            targetCoordsValue.textContent = '追跡箇所選択済み';
        } else {
            targetCoordsValue.textContent = opencvReady ? '画面をタップして選択' : '読込中...';
        }
    }

    // === Camera Enumeration ===
    async function getCameraDevices() {
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            cameraDevices = devices.filter(d => d.kind === 'videoinput');
        } catch (err) {
            console.warn('Camera enumeration failed:', err);
        }
    }

    // Apply fixed trail defaults
    tracker.setTrailParams(activePaletteColor, 6, 60, true);

    // === UI Event Bindings ===

    // Fallback: used only when OpenCV wasn't ready at tap time
    startRecordBtn.addEventListener('click', () => {
        if (!selectedPoint || !opencvReady) return;
        initiateTrackingAtPoint(selectedPoint.canvasX, selectedPoint.canvasY);
    });

    stopRecordBtn.addEventListener('click', () => {
        tracker.stopTracking();
        isRecording   = false;
        selectedPoint = null;
        stopRecordBtn.style.display  = 'none';
        startRecordBtn.style.display = 'flex';
        startRecordBtn.disabled      = true;
        recordBtnLabel.textContent   = '撮影開始';
    });

    trailResetBtn.addEventListener('click', () => {
        tracker.history       = [];
        tracker.recordingStyle = false;
    });

    // Camera picker
    function populateCameraPicker() {
        cameraPickerEl.innerHTML = '';
        if (cameraDevices.length === 0) {
            const msg = document.createElement('p');
            Object.assign(msg.style, { padding: '0.5rem 0.75rem', fontSize: '0.8rem', color: 'var(--text-secondary)' });
            msg.textContent = 'カメラが見つかりません';
            cameraPickerEl.appendChild(msg);
            return;
        }
        cameraDevices.forEach((device, index) => {
            const btn = document.createElement('button');
            btn.className = 'camera-option' + (device.deviceId === activeCameraId ? ' active' : '');
            let label = device.label || `カメラ ${index + 1}`;
            if (/back|environment|背面/i.test(label))  label = '📷 背面カメラ';
            else if (/front|user|前面/i.test(label))   label = '🤳 前面カメラ';
            btn.textContent = label;
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                startCamera(device.deviceId);
                cameraPickerEl.style.display = 'none';
            });
            cameraPickerEl.appendChild(btn);
        });
    }

    cameraSelectorBtn.addEventListener('click', async (e) => {
        e.stopPropagation();
        if (cameraPickerEl.style.display !== 'none') {
            cameraPickerEl.style.display = 'none';
            return;
        }
        await getCameraDevices();
        populateCameraPicker();
        cameraPickerEl.style.display = 'block';
    });

    document.addEventListener('click', () => {
        cameraPickerEl.style.display = 'none';
    });

    rangeExpandBtn.addEventListener('click', () => {
        tracker.searchWindowMultiplier = Math.min(tracker.searchWindowMultiplier + 0.5, 8.0);
        showToast(`追尾範囲: ${tracker.searchWindowMultiplier.toFixed(1)}x`);
    });

    rangeShrinkBtn.addEventListener('click', () => {
        tracker.searchWindowMultiplier = Math.max(tracker.searchWindowMultiplier - 0.5, 1.5);
        showToast(`追尾範囲: ${tracker.searchWindowMultiplier.toFixed(1)}x`);
    });

    // === OpenCV Callbacks ===
    const initOpenCv = function() {
        console.log('OpenCV.js onRuntimeInitialized — WASM ready');
        opencvReady = true;

        // Hide the "engine loading" badge
        if (opencvBadge) opencvBadge.style.display = 'none';

        showToast('🎯 追尾エンジン準備完了！タップして追尾開始');
        console.log('Tracking engine ready.');
    };

    window.onOpenCvReadyCallback = function() {
        cv['onRuntimeInitialized'] = initOpenCv;
        // Fallback: if WASM already compiled synchronously
        if (cv.Mat) initOpenCv();
    };

    window.onOpenCvFailCallback = function() {
        opencvFailed = true;
        if (opencvBadge) {
            opencvBadge.textContent = '⚠ エンジン読込失敗';
            opencvBadge.style.background = 'rgba(236,72,153,0.15)';
            opencvBadge.style.borderColor = 'rgba(236,72,153,0.4)';
            opencvBadge.style.color = '#ec4899';
            opencvBadge.style.animation = 'none';
        }
        showToast('⚠ 追尾エンジンの読込に失敗しました（ネット接続確認を）');
    };

    // Handle race: OpenCV may have already fired before DOMContentLoaded
    if (window.opencvReadyTriggered && typeof cv !== 'undefined') {
        window.onOpenCvReadyCallback();
    }

    // === IMMEDIATELY start camera (no OpenCV dependency) ===
    (async () => {
        await getCameraDevices();
        navigator.mediaDevices.ondevicechange = getCameraDevices;

        // Show OpenCV loading badge once camera is live
        if (opencvBadge) opencvBadge.style.display = 'block';

        await startCamera();
    })();
});
