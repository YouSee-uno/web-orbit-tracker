/**
 * Orbit Tracker - OpenCV.js Core Tracking Engine
 * Implements high-performance Localized Template Matching and HSV Color Tracking
 */

class OrbitTracker {
    constructor() {
        this.mode = 'color'; // 'template' or 'color'
        this.isTracking = false;
        this.status = 'NONE'; // 'NONE', 'TRACKED', 'LOST', 'SEARCHING'
        
        // Target positioning (in 640x480 local coordinate space)
        this.targetX = 0;
        this.targetY = 0;
        
        // Trajectory point history
        this.history = [];
        this.maxHistoryLength = 60;
        
        // Template Matching parameters
        this.templateSize = 64; // Size of square template
        this.templateW = 64;   // Actual template width (may differ from templateSize for rect selections)
        this.templateH = 64;   // Actual template height
        this.searchWindowMultiplier = 3.0; // Search window size relative to template
        this.templateMat = null; // Stored cv.Mat for tracking template
        this.matchingThreshold = 0.65; // TM_CCOEFF_NORMED minimum acceptable score
        
        // Color Tracking parameters
        this.targetHsv = null; // [H, S, V] average at tap point
        this.hTolerance = 18; // hue ±18° (wider for fast objects under motion blur)
        this.sTolerance = 60; // saturation tolerance
        this.vTolerance = 70; // value/brightness tolerance
        this.minBlobArea = 80; // minimum contour area to consider (px²)
        
        // Trail Aesthetics
        this.trailColor = '#10b981'; // Default emerald
        this.trailWidth = 6;
        this.trailFade = 1.0; // Speed of fading (1.0 = normal, 0.5 = slow)
        this.fadeEnabled = true; // Toggle for non-fading permanent path lines
        this.recordingStyle = false; // When true: draw trail at 5px / 70% opacity
    }

    /**
     * Set the current tracking mode ('template' or 'color')
     */
    setMode(mode) {
        if (this.mode !== mode) {
            this.mode = mode;
            this.reset();
        }
    }

    /**
     * Set trail appearance parameters
     */
    setTrailParams(color, width, maxHistory, fadeEnabled) {
        this.trailColor = color;
        this.trailWidth = parseFloat(width);
        this.maxHistoryLength = parseInt(maxHistory);
        this.fadeEnabled = fadeEnabled !== false;
        
        // Trim history if needed based on active fade limit
        const limit = this.fadeEnabled ? this.maxHistoryLength : 3000;
        while (this.history.length > limit) {
            this.history.shift();
        }
    }

    /**
     * Reset tracker state and release WebAssembly memory
     */
    reset() {
        this.isTracking = false;
        this.status = 'NONE';
        this.history = [];
        this.recordingStyle = false;

        // Important: Release OpenCV Mat objects to prevent Wasm memory leaks!
        if (this.templateMat) {
            this.templateMat.delete();
            this.templateMat = null;
        }

        this.targetHsv = null;
    }

    /**
     * Stop tracking but preserve the history trail for review
     */
    stopTracking() {
        this.isTracking = false;
        this.status = 'NONE';
        if (this.templateMat) {
            this.templateMat.delete();
            this.templateMat = null;
        }
        this.targetHsv = null;
    }

    /**
     * Start tracking from a coordinate tapped on screen
     * @param {cv.Mat} srcMat - Current frame matrix (RGBA, 640x480 resolution)
     * @param {number} tapX - Normalized X coordinate in srcMat space
     * @param {number} tapY - Normalized Y coordinate in srcMat space
     */
    initTracking(srcMat, tapX, tapY) {
        this.reset();
        
        // Clamp tap coordinates to frame boundaries
        const cols = srcMat.cols;
        const rows = srcMat.rows;
        this.targetX = Math.max(0, Math.min(cols - 1, tapX));
        this.targetY = Math.max(0, Math.min(rows - 1, tapY));

        if (this.mode === 'template') {
            this.initTemplateMatching(srcMat);
        } else {
            this.initColorTracking(srcMat);
        }
        
        this.isTracking = true;
        this.status = 'TRACKED';
        this.history.push({ x: this.targetX, y: this.targetY, time: Date.now() });
    }

    /**
     * Crop and store the template patch centered around tap coordinates
     */
    initTemplateMatching(srcMat) {
        const halfSize = Math.floor(this.templateSize / 2);
        
        // Define cropping rectangle box centered at target
        let x = Math.max(0, this.targetX - halfSize);
        let y = Math.max(0, this.targetY - halfSize);
        let w = this.templateSize;
        let h = this.templateSize;

        // Clip rectangle to image boundaries
        if (x + w > srcMat.cols) x = srcMat.cols - w;
        if (y + h > srcMat.rows) y = srcMat.rows - h;
        
        // Safeguard size if image is too small
        w = Math.min(w, srcMat.cols - x);
        h = Math.min(h, srcMat.rows - y);

        if (w <= 0 || h <= 0) return;

        this.templateW = w;
        this.templateH = h;

        // Crop the template Mat
        let rect = new cv.Rect(x, y, w, h);
        let cropped = srcMat.roi(rect);
        
        // Convert to grayscale for fast and illumination-robust matching
        this.templateMat = new cv.Mat();
        cv.cvtColor(cropped, this.templateMat, cv.COLOR_RGBA2GRAY);
        
        cropped.delete();
    }

    /**
     * Sample color in a small region around target coordinates and convert to HSV range
     */
    initColorTracking(srcMat) {
        // Convert the full frame to HSV first
        let hsvMat = new cv.Mat();
        cv.cvtColor(srcMat, hsvMat, cv.COLOR_RGBA2RGB); // cvtColor requires RGB for HSV conversion
        cv.cvtColor(hsvMat, hsvMat, cv.COLOR_RGB2HSV);
        
        // Sample HSV values in a 5x5 neighborhood to reduce single-pixel noise
        const sampleRadius = 2;
        let sumH = 0, sumS = 0, sumV = 0, count = 0;
        
        for (let dy = -sampleRadius; dy <= sampleRadius; dy++) {
            for (let dx = -sampleRadius; dx <= sampleRadius; dx++) {
                const px = Math.floor(this.targetX + dx);
                const py = Math.floor(this.targetY + dy);
                
                if (px >= 0 && px < hsvMat.cols && py >= 0 && py < hsvMat.rows) {
                    const pixel = hsvMat.ucharPtr(py, px);
                    sumH += pixel[0];
                    sumS += pixel[1];
                    sumV += pixel[2];
                    count++;
                }
            }
        }
        
        hsvMat.delete();

        if (count > 0) {
            this.targetHsv = [
                Math.round(sumH / count),
                Math.round(sumS / count),
                Math.round(sumV / count)
            ];
        } else {
            this.targetHsv = [0, 0, 0];
        }
    }

    /**
     * Start tracking from a user-drawn rectangle region
     * @param {cv.Mat} srcMat - Current frame matrix (RGBA, 640x480)
     * @param {number} rx - Region left edge in process-space coordinates
     * @param {number} ry - Region top edge in process-space coordinates
     * @param {number} rw - Region width
     * @param {number} rh - Region height
     */
    initTrackingFromRect(srcMat, rx, ry, rw, rh) {
        this.reset();

        rx = Math.max(0, Math.floor(rx));
        ry = Math.max(0, Math.floor(ry));
        rw = Math.min(Math.floor(rw), srcMat.cols - rx);
        rh = Math.min(Math.floor(rh), srcMat.rows - ry);

        if (rw <= 0 || rh <= 0) return;

        this.targetX = rx + rw / 2;
        this.targetY = ry + rh / 2;
        this.templateSize = Math.max(rw, rh); // used for search window sizing

        if (this.mode === 'template') {
            this.templateW = rw;
            this.templateH = rh;

            let rect = new cv.Rect(rx, ry, rw, rh);
            let cropped = srcMat.roi(rect);
            this.templateMat = new cv.Mat();
            cv.cvtColor(cropped, this.templateMat, cv.COLOR_RGBA2GRAY);
            cropped.delete();
        } else {
            this.initColorTracking(srcMat);
        }

        this.isTracking = true;
        this.status = 'TRACKED';
        this.history.push({ x: this.targetX, y: this.targetY, time: Date.now() });
    }

    /**
     * Main tracking update per frame
     * @param {cv.Mat} srcMat - Current frame matrix (RGBA, 640x480)
     */
    processFrame(srcMat) {
        if (!this.isTracking) return;

        let trackingResult = null;
        
        if (this.mode === 'template') {
            trackingResult = this.trackTemplate(srcMat);
        } else {
            trackingResult = this.trackColor(srcMat);
        }

        if (trackingResult && trackingResult.status === 'TRACKED') {
            this.targetX = trackingResult.x;
            this.targetY = trackingResult.y;
            this.status = 'TRACKED';

            // Add to history trail
            this.history.push({ x: this.targetX, y: this.targetY, time: Date.now() });

            // Recording mode: never trim — trail grows until stop is pressed
            if (!this.recordingStyle) {
                const limit = this.fadeEnabled ? this.maxHistoryLength : 3000;
                while (this.history.length > limit) {
                    this.history.shift();
                }
            }
        } else {
            this.status = 'LOST';
            // Recording mode: keep trail intact when lost — do NOT remove points
            if (!this.recordingStyle && this.fadeEnabled && this.history.length > 0) {
                this.history.shift();
            }
        }
    }

    /**
     * Localized Template Matching Tracker
     */
    trackTemplate(srcMat) {
        if (!this.templateMat || this.templateMat.empty()) return { status: 'LOST' };

        const frameWidth = srcMat.cols;
        const frameHeight = srcMat.rows;
        const tW = this.templateW;
        const tH = this.templateH;

        // 1. Define localized Search Window centered around last tracked position
        const searchSize = Math.floor(this.templateSize * this.searchWindowMultiplier);
        const halfSSize = Math.floor(searchSize / 2);
        
        let sx = Math.max(0, this.targetX - halfSSize);
        let sy = Math.max(0, this.targetY - halfSSize);
        let sw = searchSize;
        let sh = searchSize;

        // Adjust search window boundaries
        if (sx + sw > frameWidth) sx = frameWidth - sw;
        if (sy + sh > frameHeight) sy = frameHeight - sh;
        sw = Math.min(sw, frameWidth - sx);
        sh = Math.min(sh, frameHeight - sy);

        // Ensure search window is larger than the template
        if (sw < tW || sh < tH) {
            return { status: 'LOST' };
        }

        // 2. Crop Search Window and convert to Grayscale
        let searchRect = new cv.Rect(sx, sy, sw, sh);
        let searchRoi = srcMat.roi(searchRect);
        
        let searchGray = new cv.Mat();
        cv.cvtColor(searchRoi, searchGray, cv.COLOR_RGBA2GRAY);
        
        // 3. Perform Template Matching inside search region
        let resultMat = new cv.Mat();
        let mask = new cv.Mat(); // Unused
        
        cv.matchTemplate(searchGray, this.templateMat, resultMat, cv.TM_CCOEFF_NORMED, mask);
        
        // 4. Find location of the peak match
        let minMax = cv.minMaxLoc(resultMat);
        let maxVal = minMax.maxVal;
        let maxLoc = minMax.maxLoc;

        // 5. Cleanup local Wasm matrices
        searchRoi.delete();
        searchGray.delete();
        resultMat.delete();
        mask.delete();

        // 6. Verify match strength
        if (maxVal >= this.matchingThreshold) {
            // Match location is top-left corner of the template inside search window.
            const newCenterX = sx + maxLoc.x + tW / 2;
            const newCenterY = sy + maxLoc.y + tH / 2;

            return {
                x: newCenterX,
                y: newCenterY,
                score: maxVal,
                status: 'TRACKED'
            };
        } else {
            return { status: 'LOST' };
        }
    }

    /**
     * Color Range contour centroid tracking
     */
    trackColor(srcMat) {
        if (!this.targetHsv) return { status: 'LOST' };

        // 1. Convert source frame to HSV format
        let rgbMat = new cv.Mat();
        cv.cvtColor(srcMat, rgbMat, cv.COLOR_RGBA2RGB);
        
        let hsvMat = new cv.Mat();
        cv.cvtColor(rgbMat, hsvMat, cv.COLOR_RGB2HSV);
        rgbMat.delete();

        // 2. Establish HSV thresholds with wrap-around support for hue [0-180]
        const h = this.targetHsv[0];
        const s = this.targetHsv[1];
        const v = this.targetHsv[2];

        let mask = new cv.Mat();

        // Check if Hue color bounds wrap around 0/180 boundaries (common for reds)
        let hLower1 = Math.max(0, h - this.hTolerance);
        let hUpper1 = Math.min(180, h + this.hTolerance);
        
        let sLower = Math.max(10, s - this.sTolerance); // Minimum saturation of 10 to filter grays
        let sUpper = Math.min(255, s + this.sTolerance);
        
        let vLower = Math.max(20, v - this.vTolerance); // Minimum value of 20 to filter dark shadows
        let vUpper = Math.min(255, v + this.vTolerance);

        let low1 = new cv.Mat(hsvMat.rows, hsvMat.cols, hsvMat.type(), [hLower1, sLower, vLower, 0]);
        let high1 = new cv.Mat(hsvMat.rows, hsvMat.cols, hsvMat.type(), [hUpper1, sUpper, vUpper, 0]);
        cv.inRange(hsvMat, low1, high1, mask);
        low1.delete();
        high1.delete();

        // Handle Red color wrap-around logic
        if (h - this.hTolerance < 0) {
            let hLower2 = 180 + (h - this.hTolerance);
            let hUpper2 = 180;
            let low2 = new cv.Mat(hsvMat.rows, hsvMat.cols, hsvMat.type(), [hLower2, sLower, vLower, 0]);
            let high2 = new cv.Mat(hsvMat.rows, hsvMat.cols, hsvMat.type(), [hUpper2, sUpper, vUpper, 0]);
            
            let tempMask = new cv.Mat();
            cv.inRange(hsvMat, low2, high2, tempMask);
            
            // Merge both masks using bitwise OR
            cv.bitwise_or(mask, tempMask, mask);
            
            low2.delete();
            high2.delete();
            tempMask.delete();
        } else if (h + this.hTolerance > 180) {
            let hLower2 = 0;
            let hUpper2 = (h + this.hTolerance) - 180;
            let low2 = new cv.Mat(hsvMat.rows, hsvMat.cols, hsvMat.type(), [hLower2, sLower, vLower, 0]);
            let high2 = new cv.Mat(hsvMat.rows, hsvMat.cols, hsvMat.type(), [hUpper2, sUpper, vUpper, 0]);
            
            let tempMask = new cv.Mat();
            cv.inRange(hsvMat, low2, high2, tempMask);
            
            cv.bitwise_or(mask, tempMask, mask);
            
            low2.delete();
            high2.delete();
            tempMask.delete();
        }

        hsvMat.delete();

        // 3. Noise filtering using morphological opening/closing
        let kSize = new cv.Size(5, 5);
        let M = cv.getStructuringElement(cv.MORPH_RECT, kSize);
        let anchor = new cv.Point(-1, -1);
        
        cv.morphologyEx(mask, mask, cv.MORPH_OPEN, M, anchor, 1);
        cv.morphologyEx(mask, mask, cv.MORPH_CLOSE, M, anchor, 1);
        M.delete();

        // 4. Find all contours in binary mask
        let contours = new cv.MatVector();
        let hierarchy = new cv.Mat();
        cv.findContours(mask, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

        let bestCentroid = null;
        let bestDist = Infinity;

        // Pick the contour closest to the last known position.
        // This handles fast objects far better than "largest area" because the
        // correct blob stays near the previous position even when other same-colour
        // regions exist in the frame.
        for (let i = 0; i < contours.size(); ++i) {
            let cnt = contours.get(i);
            let area = cv.contourArea(cnt);

            if (area > this.minBlobArea) {
                let m = cv.moments(cnt);
                if (m.m00 !== 0) {
                    const cx = Math.round(m.m10 / m.m00);
                    const cy = Math.round(m.m01 / m.m00);
                    const dx = cx - this.targetX;
                    const dy = cy - this.targetY;
                    const dist = dx * dx + dy * dy; // squared — no sqrt needed for comparison

                    if (dist < bestDist) {
                        bestDist = dist;
                        bestCentroid = { x: cx, y: cy };
                    }
                }
            }
        }

        // Cleanup
        mask.delete();
        contours.delete();
        hierarchy.delete();

        if (bestCentroid) {
            return {
                x: bestCentroid.x,
                y: bestCentroid.y,
                status: 'TRACKED'
            };
        } else {
            return { status: 'LOST' };
        }
    }

    /**
     * Render the trajectory trail and target reticle to HTML Canvas
     * @param {CanvasRenderingContext2D} ctx - Main Canvas 2D Context
     * @param {number} canvasWidth - Scale coordinate from 640x480 space
     * @param {number} canvasHeight - Scale coordinate from 640x480 space
     */
    draw(ctx, canvasWidth, canvasHeight) {
        if (this.history.length < 1) return;

        // Scaling factors from internal 640x480 to screen canvas size
        const scaleX = canvasWidth / 640;
        const scaleY = canvasHeight / 480;

        // 1. Draw glowing comet trail (connecting point history)
        if (this.history.length > 1) {
            ctx.save();
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';

            if (this.recordingStyle) {
                // Recording mode: fixed 5px width, 70% opacity solid line
                ctx.beginPath();
                ctx.moveTo(this.history[0].x * scaleX, this.history[0].y * scaleY);
                for (let i = 1; i < this.history.length; i++) {
                    ctx.lineTo(this.history[i].x * scaleX, this.history[i].y * scaleY);
                }
                ctx.strokeStyle = this.trailColor;
                ctx.globalAlpha = 0.7;
                ctx.lineWidth = 5;
                ctx.stroke();
            } else if (this.fadeEnabled) {
                // Fading comet tail mode (draw segment by segment)
                for (let i = 1; i < this.history.length; i++) {
                    const p1 = this.history[i - 1];
                    const p2 = this.history[i];

                    const ratio = i / this.history.length;
                    const alpha = ratio * 0.9;
                    const width = ratio * this.trailWidth;

                    // Glowing outer effect
                    ctx.beginPath();
                    ctx.moveTo(p1.x * scaleX, p1.y * scaleY);
                    ctx.lineTo(p2.x * scaleX, p2.y * scaleY);
                    
                    ctx.strokeStyle = this.trailColor;
                    ctx.globalAlpha = alpha * 0.3;
                    ctx.lineWidth = width * 2.2;
                    ctx.stroke();

                    // Bright core line
                    ctx.beginPath();
                    ctx.moveTo(p1.x * scaleX, p1.y * scaleY);
                    ctx.lineTo(p2.x * scaleX, p2.y * scaleY);
                    
                    ctx.strokeStyle = '#ffffff';
                    if (this.trailColor !== '#ffffff') {
                        ctx.strokeStyle = this.trailColor;
                    }
                    ctx.globalAlpha = alpha;
                    ctx.lineWidth = width;
                    ctx.stroke();
                }
            } else {
                // Permanent Solid trajectory line mode (efficient continuous path)
                // 1. Thicker outer neon glow pass
                ctx.beginPath();
                ctx.moveTo(this.history[0].x * scaleX, this.history[0].y * scaleY);
                for (let i = 1; i < this.history.length; i++) {
                    ctx.lineTo(this.history[i].x * scaleX, this.history[i].y * scaleY);
                }
                ctx.strokeStyle = this.trailColor;
                ctx.globalAlpha = 0.35;
                ctx.lineWidth = this.trailWidth * 2.2;
                ctx.stroke();

                // 2. Main bright core pass
                ctx.beginPath();
                ctx.moveTo(this.history[0].x * scaleX, this.history[0].y * scaleY);
                for (let i = 1; i < this.history.length; i++) {
                    ctx.lineTo(this.history[i].x * scaleX, this.history[i].y * scaleY);
                }
                ctx.strokeStyle = this.trailColor;
                ctx.globalAlpha = 0.95;
                ctx.lineWidth = this.trailWidth;
                ctx.stroke();
            }
            ctx.restore();
        }

        // 2. Draw Target Reticle (only if active tracking)
        if (this.isTracking) {
            ctx.save();
            const tx = this.targetX * scaleX;
            const ty = this.targetY * scaleY;
            
            // Reticle color and animation state
            let pulseSize = 25 + Math.sin(Date.now() / 150) * 3;
            
            if (this.status === 'TRACKED') {
                ctx.strokeStyle = this.trailColor;
                ctx.fillStyle = this.trailColor;
                ctx.shadowColor = this.trailColor;
                ctx.shadowBlur = 10;
            } else {
                // Lost tracking: red flashing reticle
                ctx.strokeStyle = '#ec4899'; // magenta
                ctx.fillStyle = '#ec4899';
                ctx.shadowColor = '#ec4899';
                ctx.shadowBlur = 15;
                pulseSize = 30 + Math.sin(Date.now() / 80) * 5;
            }

            // Draw target center dot
            ctx.beginPath();
            ctx.arc(tx, ty, 3, 0, Math.PI * 2);
            ctx.fill();

            // Draw outer tracking bounding reticle circle with corner tick-marks
            ctx.beginPath();
            ctx.arc(tx, ty, pulseSize, 0, Math.PI * 2);
            ctx.lineWidth = 1.5;
            if (this.status === 'LOST') {
                ctx.setLineDash([4, 4]); // dashed if lost
            }
            ctx.stroke();

            // Draw four tick marks pointing towards the center
            ctx.lineWidth = 2;
            ctx.setLineDash([]); // clear dash

            const tickLen = 8;
            const r = pulseSize;

            // Top
            ctx.beginPath(); ctx.moveTo(tx, ty - r - 2); ctx.lineTo(tx, ty - r + tickLen); ctx.stroke();
            // Bottom
            ctx.beginPath(); ctx.moveTo(tx, ty + r + 2); ctx.lineTo(tx, ty + r - tickLen); ctx.stroke();
            // Left
            ctx.beginPath(); ctx.moveTo(tx - r - 2, ty); ctx.lineTo(tx - r + tickLen, ty); ctx.stroke();
            // Right
            ctx.beginPath(); ctx.moveTo(tx + r + 2, ty); ctx.lineTo(tx + r - tickLen, ty); ctx.stroke();

            // Draw tag label above reticle
            ctx.shadowBlur = 0; // disable shadow for text sharpness
            ctx.fillStyle = '#ffffff';
            ctx.font = 'bold 11px Outfit, sans-serif';
            ctx.textAlign = 'center';
            
            const modeText = this.mode === 'template' ? 'TARGET [PATTERN]' : 'TARGET [COLOR]';
            const statusLabel = this.status === 'TRACKED' ? modeText : 'SIGNAL LOST';
            
            ctx.fillText(statusLabel, tx, ty - r - 10);

            ctx.restore();
        }
    }
}

// Attach to window object to share scope with app.js
window.OrbitTracker = OrbitTracker;
