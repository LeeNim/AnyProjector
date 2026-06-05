/**
 * AnyProjector Dataset Recorder
 *
 * Thu âm → Augmentation → Annotation → Export ZIP
 * Sử dụng Web Audio API + MediaRecorder API
 */

// ============================================
// State
// ============================================
const state = {
    /** @type {MediaStream|null} */
    stream: null,
    /** @type {MediaRecorder|null} */
    recorder: null,
    /** @type {AudioContext|null} */
    audioCtx: null,
    /** @type {AnalyserNode|null} */
    analyser: null,

    deviceId: '',
    sampleCount: 10,
    phase: 'phase2_alignment',

    /** @type {{blob: Blob, url: string, filename: string, isAugmented: boolean, augType: string|null, parentIdx: number|null}[]} */
    samples: [],
    /**
     * Agnostic annotation format:
     * Phase 2: { transcript: string }
     * Phase 3: { transcript: string, output: { type: 'text'|'tool_call'|'mixed', content?: string, calls?: [{name, args}] } }
     * @type {Object[]}
     */
    annotations: [],

    currentSampleIdx: 0,
    isRecording: false,
    timerInterval: null,
    animationFrame: null,
    RECORD_DURATION: 10, // seconds, fixed
};

// ============================================
// DOM refs
// ============================================
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const DOM = {
    // Config
    audioDevice: $('#audio-device'),
    sampleCount: $('#sample-count'),
    datasetPhase: $('#dataset-phase'),
    startSessionBtn: $('#start-session-btn'),

    // Recording
    configSection: $('#config-section'),
    recordingSection: $('#recording-section'),
    progressText: $('#progress-text'),
    progressBar: $('#progress-bar'),
    canvas: $('#waveform-canvas'),
    timer: $('#timer'),
    recordBtn: $('#record-btn'),
    stopBtn: $('#stop-btn'),
    currentSampleNum: $('#current-sample-num'),
    samplesList: $('#samples-list'),
    recordingActions: $('#recording-actions'),
    goToAugmentBtn: $('#go-to-augment-btn'),

    // Augment
    augmentSection: $('#augment-section'),
    augPitchUp: $('#aug-pitch-up'),
    augPitchDown: $('#aug-pitch-down'),
    augNoise: $('#aug-noise'),
    augmentSummary: $('#augment-summary'),
    augCount: $('#aug-count'),
    skipAugmentBtn: $('#skip-augment-btn'),
    runAugmentBtn: $('#run-augment-btn'),

    // Annotate
    annotateSection: $('#annotate-section'),
    annotateProgressText: $('#annotate-progress-text'),
    annotateList: $('#annotate-list'),
    goToExportBtn: $('#go-to-export-btn'),

    // Export
    exportSection: $('#export-section'),
    exportPreview: $('#export-preview'),
    exportBtn: $('#export-btn'),
};

// ============================================
// Init
// ============================================
async function init() {
    await loadAudioDevices();
    bindEvents();
}

async function loadAudioDevices() {
    try {
        // Need permission first to get device labels
        const tempStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        tempStream.getTracks().forEach(t => t.stop());

        const devices = await navigator.mediaDevices.enumerateDevices();
        const audioInputs = devices.filter(d => d.kind === 'audioinput');

        DOM.audioDevice.innerHTML = '';
        audioInputs.forEach((device, i) => {
            const opt = document.createElement('option');
            opt.value = device.deviceId;
            opt.textContent = device.label || `Microphone ${i + 1}`;
            DOM.audioDevice.appendChild(opt);
        });

        if (audioInputs.length === 0) {
            DOM.audioDevice.innerHTML = '<option value="">Không tìm thấy microphone</option>';
        }
    } catch (err) {
        DOM.audioDevice.innerHTML = '<option value="">Lỗi: cần cấp quyền microphone</option>';
        console.error('Failed to enumerate audio devices:', err);
    }
}

function bindEvents() {
    DOM.startSessionBtn.addEventListener('click', startSession);
    DOM.recordBtn.addEventListener('click', startRecording);
    DOM.stopBtn.addEventListener('click', stopRecording);
    DOM.goToAugmentBtn.addEventListener('click', goToAugment);
    DOM.skipAugmentBtn.addEventListener('click', skipAugment);
    DOM.runAugmentBtn.addEventListener('click', runAugmentation);
    DOM.goToExportBtn.addEventListener('click', goToExport);
    DOM.exportBtn.addEventListener('click', exportDataset);

    // Update aug count preview
    [DOM.augPitchUp, DOM.augPitchDown, DOM.augNoise].forEach(el => {
        el.addEventListener('change', updateAugCount);
    });
}

// ============================================
// Step 1: Start Session
// ============================================
function startSession() {
    state.deviceId = DOM.audioDevice.value;
    state.sampleCount = parseInt(DOM.sampleCount.value) || 10;
    state.phase = DOM.datasetPhase.value;

    if (!state.deviceId) {
        alert('Vui lòng chọn thiết bị thu âm.');
        return;
    }

    state.samples = [];
    state.annotations = [];
    state.currentSampleIdx = 0;

    // Show recording section
    DOM.recordingSection.classList.remove('hidden');
    DOM.recordingSection.scrollIntoView({ behavior: 'smooth' });

    updateProgress();
    updateCurrentSampleNum();
}

// ============================================
// Step 2: Recording
// ============================================
async function startRecording() {
    try {
        state.stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                deviceId: { exact: state.deviceId },
                sampleRate: 16000,
                channelCount: 1,
                echoCancellation: false,
                noiseSuppression: false,
                autoGainControl: false,
            }
        });
    } catch (err) {
        console.error('getUserMedia failed:', err);
        alert('Không thể truy cập microphone. Kiểm tra quyền truy cập.');
        return;
    }

    // Setup audio context for waveform visualization
    state.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const source = state.audioCtx.createMediaStreamSource(state.stream);
    state.analyser = state.audioCtx.createAnalyser();
    state.analyser.fftSize = 2048;
    source.connect(state.analyser);

    // Setup MediaRecorder
    const chunks = [];
    state.recorder = new MediaRecorder(state.stream, { mimeType: 'audio/webm;codecs=opus' });

    state.recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.push(e.data);
    };

    state.recorder.onstop = () => {
        const blob = new Blob(chunks, { type: 'audio/webm' });
        const url = URL.createObjectURL(blob);
        const idx = state.currentSampleIdx;
        const filename = `sample_${String(idx + 1).padStart(3, '0')}.wav`;

        // Replace or add
        if (idx < state.samples.length && !state.samples[idx].isAugmented) {
            URL.revokeObjectURL(state.samples[idx].url);
            state.samples[idx] = { blob, url, filename, isAugmented: false, augType: null, parentIdx: null };
        } else {
            state.samples.push({ blob, url, filename, isAugmented: false, augType: null, parentIdx: null });
        }

        renderSampleItem(idx);
        state.currentSampleIdx = getNextOriginalIdx();
        updateProgress();
        updateCurrentSampleNum();
        checkAllRecorded();

        // Cleanup
        cleanup();
    };

    // Start recording
    state.recorder.start();
    state.isRecording = true;

    // UI
    DOM.recordBtn.classList.add('recording');
    DOM.recordBtn.disabled = true;
    DOM.stopBtn.classList.remove('hidden');
    DOM.timer.classList.add('active');

    // Timer countdown
    let remaining = state.RECORD_DURATION;
    DOM.timer.textContent = remaining.toFixed(1);

    state.timerInterval = setInterval(() => {
        remaining -= 0.1;
        DOM.timer.textContent = Math.max(0, remaining).toFixed(1);

        if (remaining <= 0) {
            stopRecording();
        }
    }, 100);

    // Waveform
    drawWaveform();
}

function stopRecording() {
    if (!state.isRecording) return;
    state.isRecording = false;

    clearInterval(state.timerInterval);
    cancelAnimationFrame(state.animationFrame);

    if (state.recorder && state.recorder.state === 'recording') {
        state.recorder.stop();
    }

    // UI reset
    DOM.recordBtn.classList.remove('recording');
    DOM.recordBtn.disabled = false;
    DOM.stopBtn.classList.add('hidden');
    DOM.timer.classList.remove('active');
    DOM.timer.textContent = state.RECORD_DURATION.toFixed(1);
}

function cleanup() {
    if (state.stream) {
        state.stream.getTracks().forEach(t => t.stop());
        state.stream = null;
    }
    if (state.audioCtx) {
        state.audioCtx.close();
        state.audioCtx = null;
    }
}

function drawWaveform() {
    const canvas = DOM.canvas;
    const ctx = canvas.getContext('2d');
    const analyser = state.analyser;

    if (!analyser || !state.isRecording) return;

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    function draw() {
        if (!state.isRecording) return;
        state.animationFrame = requestAnimationFrame(draw);

        analyser.getByteTimeDomainData(dataArray);

        ctx.fillStyle = '#14141c';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        ctx.lineWidth = 2;
        ctx.strokeStyle = '#6366f1';
        ctx.beginPath();

        const sliceWidth = canvas.width / bufferLength;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
            const v = dataArray[i] / 128.0;
            const y = (v * canvas.height) / 2;

            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);

            x += sliceWidth;
        }

        ctx.lineTo(canvas.width, canvas.height / 2);
        ctx.stroke();
    }

    draw();
}

function getNextOriginalIdx() {
    const originalCount = state.samples.filter(s => !s.isAugmented).length;
    return originalCount;
}

function renderSampleItem(idx) {
    const sample = state.samples[idx];
    let item = document.getElementById(`sample-${idx}`);

    if (!item) {
        item = document.createElement('div');
        item.className = 'sample-item';
        item.id = `sample-${idx}`;
        DOM.samplesList.appendChild(item);
    }

    item.innerHTML = `
        <span class="sample-num">#${idx + 1}</span>
        <div class="sample-audio">
            <audio controls src="${sample.url}" preload="metadata"></audio>
        </div>
        <div class="sample-actions">
            <button class="btn btn-secondary btn-sm" onclick="reRecord(${idx})">🔄 Thu lại</button>
        </div>
        <span class="sample-status">✓</span>
    `;
}

function reRecord(idx) {
    if (state.isRecording) return;
    state.currentSampleIdx = idx;
    updateCurrentSampleNum();

    // Re-enable record button (may have been disabled by checkAllRecorded)
    DOM.recordBtn.disabled = false;
    DOM.recordBtn.scrollIntoView({ behavior: 'smooth' });
}

function updateProgress() {
    const originalCount = state.samples.filter(s => !s.isAugmented).length;
    DOM.progressText.textContent = `Mẫu ${originalCount}/${state.sampleCount}`;
    DOM.progressBar.style.width = `${(originalCount / state.sampleCount) * 100}%`;
}

function updateCurrentSampleNum() {
    DOM.currentSampleNum.textContent = state.currentSampleIdx + 1;
}

function checkAllRecorded() {
    const originalCount = state.samples.filter(s => !s.isAugmented).length;
    if (originalCount >= state.sampleCount) {
        DOM.recordingActions.classList.remove('hidden');
        DOM.recordBtn.disabled = true;
    } else {
        DOM.recordBtn.disabled = false;
    }
}

// ============================================
// Step 3: Augmentation
// ============================================
function goToAugment() {
    DOM.augmentSection.classList.remove('hidden');
    DOM.augmentSection.scrollIntoView({ behavior: 'smooth' });
    updateAugCount();
}

function updateAugCount() {
    const original = state.samples.filter(s => !s.isAugmented).length;
    let multiplier = 0;
    if (DOM.augPitchUp.checked) multiplier++;
    if (DOM.augPitchDown.checked) multiplier++;
    if (DOM.augNoise.checked) multiplier++;

    const count = original * multiplier;
    DOM.augCount.textContent = count;
    DOM.augmentSummary.classList.toggle('hidden', count === 0);
}

function skipAugment() {
    goToAnnotate();
}

async function runAugmentation() {
    const originalSamples = state.samples.filter(s => !s.isAugmented);
    const doPitchUp = DOM.augPitchUp.checked;
    const doPitchDown = DOM.augPitchDown.checked;
    const doNoise = DOM.augNoise.checked;

    DOM.runAugmentBtn.disabled = true;
    DOM.runAugmentBtn.textContent = '⏳ Đang xử lý...';

    for (let i = 0; i < originalSamples.length; i++) {
        const sample = originalSamples[i];
        const parentIdx = state.samples.indexOf(sample);

        const audioBuffer = await decodeAudioBlob(sample.blob);

        if (doPitchUp) {
            const augmented = await pitchShift(audioBuffer, 2);
            const blob = audioBufferToWavBlob(augmented);
            const url = URL.createObjectURL(blob);
            const filename = sample.filename.replace('.wav', '_pitch_up.wav');
            state.samples.push({ blob, url, filename, isAugmented: true, augType: 'pitch_up', parentIdx });
        }

        if (doPitchDown) {
            const augmented = await pitchShift(audioBuffer, -2);
            const blob = audioBufferToWavBlob(augmented);
            const url = URL.createObjectURL(blob);
            const filename = sample.filename.replace('.wav', '_pitch_down.wav');
            state.samples.push({ blob, url, filename, isAugmented: true, augType: 'pitch_down', parentIdx });
        }

        if (doNoise) {
            const augmented = addWhiteNoise(audioBuffer, 0.01);
            const blob = audioBufferToWavBlob(augmented);
            const url = URL.createObjectURL(blob);
            const filename = sample.filename.replace('.wav', '_noisy.wav');
            state.samples.push({ blob, url, filename, isAugmented: true, augType: 'noise', parentIdx });
        }
    }

    DOM.runAugmentBtn.textContent = `✅ Đã tạo ${state.samples.length - originalSamples.length} mẫu augmented`;

    setTimeout(() => goToAnnotate(), 800);
}

// ===== Audio Processing Utils =====

async function decodeAudioBlob(blob) {
    const ctx = new OfflineAudioContext(1, 16000 * state.RECORD_DURATION, 16000);
    const arrayBuffer = await blob.arrayBuffer();
    return await ctx.decodeAudioData(arrayBuffer);
}

function pitchShift(audioBuffer, semitones) {
    // Simple pitch shift by resampling — changes speed too but
    // acceptable for training data augmentation
    const rate = Math.pow(2, semitones / 12);
    const newLength = Math.round(audioBuffer.length / rate);
    const ctx = new OfflineAudioContext(1, newLength, audioBuffer.sampleRate);

    const source = ctx.createBufferSource();
    source.buffer = audioBuffer;
    source.playbackRate.value = rate;
    source.connect(ctx.destination);
    source.start();

    // We can't await OfflineAudioContext.startRendering synchronously in all cases,
    // so we return a promise-like approach. But since this function is called in async,
    // we wrap the rendering.
    return ctx.startRendering();
}

function addWhiteNoise(audioBuffer, amplitude) {
    const ctx = new OfflineAudioContext(
        audioBuffer.numberOfChannels,
        audioBuffer.length,
        audioBuffer.sampleRate
    );

    // Copy original data
    const newBuffer = ctx.createBuffer(
        audioBuffer.numberOfChannels,
        audioBuffer.length,
        audioBuffer.sampleRate
    );

    for (let ch = 0; ch < audioBuffer.numberOfChannels; ch++) {
        const input = audioBuffer.getChannelData(ch);
        const output = newBuffer.getChannelData(ch);
        for (let i = 0; i < input.length; i++) {
            output[i] = input[i] + (Math.random() * 2 - 1) * amplitude;
        }
    }

    return newBuffer;
}

function audioBufferToWavBlob(audioBuffer) {
    // Handle both AudioBuffer and Promise<AudioBuffer>
    if (audioBuffer instanceof Promise) {
        // This shouldn't happen in our flow since we await, but safety check
        console.warn('audioBufferToWavBlob received a Promise');
        return audioBuffer.then(buf => audioBufferToWavBlob(buf));
    }

    const numChannels = audioBuffer.numberOfChannels;
    const sampleRate = audioBuffer.sampleRate;
    const format = 1; // PCM
    const bitsPerSample = 16;

    let data;
    if (numChannels === 1) {
        data = audioBuffer.getChannelData(0);
    } else {
        // Interleave channels
        const length = audioBuffer.length * numChannels;
        data = new Float32Array(length);
        for (let i = 0; i < audioBuffer.length; i++) {
            for (let ch = 0; ch < numChannels; ch++) {
                data[i * numChannels + ch] = audioBuffer.getChannelData(ch)[i];
            }
        }
    }

    const byteRate = sampleRate * numChannels * (bitsPerSample / 8);
    const blockAlign = numChannels * (bitsPerSample / 8);
    const dataSize = data.length * (bitsPerSample / 8);

    const buffer = new ArrayBuffer(44 + dataSize);
    const view = new DataView(buffer);

    // WAV header
    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + dataSize, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, format, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, byteRate, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitsPerSample, true);
    writeString(view, 36, 'data');
    view.setUint32(40, dataSize, true);

    // Write samples
    let offset = 44;
    for (let i = 0; i < data.length; i++) {
        const sample = Math.max(-1, Math.min(1, data[i]));
        view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
        offset += 2;
    }

    return new Blob([buffer], { type: 'audio/wav' });
}

function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
}

// ============================================
// Step 4: Annotation — Agnostic Format
// ============================================
//
// Format trung lập (không phụ thuộc LLM):
// Phase 2: { transcript: "..." }
// Phase 3: { transcript: "...", output: { type, content?, calls? } }
//
// Khi train, DataLoader sẽ convert sang format LLM cụ thể
// qua tokenizer.apply_chat_template()
//

function goToAnnotate() {
    DOM.annotateSection.classList.remove('hidden');
    DOM.annotateSection.scrollIntoView({ behavior: 'smooth' });

    const isPhase3 = state.phase === 'phase3_agentic';

    // Init annotations with agnostic structure
    state.annotations = state.samples.map(() => {
        const base = { transcript: '' };
        if (isPhase3) {
            base.output = { type: 'text', content: '', calls: [] };
        }
        return base;
    });

    renderAnnotateList();
    updateAnnotateProgress();
}

function renderAnnotateList() {
    DOM.annotateList.innerHTML = '';

    const isPhase3 = state.phase === 'phase3_agentic';

    state.samples.forEach((sample, idx) => {
        const item = document.createElement('div');
        item.className = 'annotate-item';
        item.id = `annotate-${idx}`;

        const augTag = sample.isAugmented
            ? `<span class="annotate-tag">${sample.augType}</span>`
            : '';

        const parentNote = sample.isAugmented
            ? `<span style="font-size:0.75rem;color:var(--text-muted)">  (t\u1EEB m\u1EABu #${sample.parentIdx + 1})</span>`
            : '';

        // Phase 3: structured output fields
        const outputFields = isPhase3
            ? `<div class="output-section">
                    <div class="form-group">
                        <label>\uD83C\uDFAF Lo\u1EA1i ph\u1EA3n h\u1ED3i</label>
                        <select id="output-type-${idx}" onchange="onOutputTypeChange(${idx})">
                            <option value="text">\uD83D\uDCAC Text (tr\u1EA3 l\u1EDDi b\u1EB1ng v\u0103n b\u1EA3n)</option>
                            <option value="tool_call">\uD83D\uDD27 Tool Call (g\u1ECDi h\u00E0m tr\u1EF1c ti\u1EBFp)</option>
                            <option value="mixed">\uD83E\uDDE0 Suy lu\u1EADn + Tool Call</option>
                        </select>
                    </div>
                    <div class="form-group" id="output-content-group-${idx}">
                        <label>\uD83D\uDCDD N\u1ED9i dung text</label>
                        <textarea
                            id="output-content-${idx}"
                            placeholder="VD: C\u1EA3m \u01A1n b\u1EA1n! T\u00F4i lu\u00F4n s\u1EB5n s\u00E0ng h\u1ED7 tr\u1EE3."
                            oninput="onAnnotationChange(${idx})"
                        ></textarea>
                    </div>
                    <div class="tool-call-section hidden" id="tool-section-${idx}">
                        <div class="form-group">
                            <label>\uD83D\uDD27 T\u00EAn function</label>
                            <input type="text" id="tool-name-${idx}"
                                placeholder="VD: turn_on_light, get_weather, play_music"
                                oninput="onAnnotationChange(${idx})">
                        </div>
                        <div class="form-group">
                            <label>\uD83D\uDCE6 Arguments (JSON)</label>
                            <textarea
                                id="tool-args-${idx}"
                                placeholder='{"room": "ph\u00F2ng kh\u00E1ch"}'
                                oninput="onAnnotationChange(${idx})"
                                style="font-family: monospace; min-height: 48px;"
                            ></textarea>
                        </div>
                    </div>
               </div>`
            : '';

        item.innerHTML = `
            <div class="annotate-item-header">
                <span class="sample-num">#${idx + 1}</span>
                ${augTag}
                ${parentNote}
                <audio controls src="${sample.url}" preload="metadata"></audio>
            </div>
            <div class="annotate-fields">
                <div class="form-group">
                    <label>\uD83D\uDCDD Transcript (phi\u00EAn \u00E2m)</label>
                    <textarea
                        id="transcript-${idx}"
                        placeholder="Phi\u00EAn \u00E2m: b\u1EA1n n\u00F3i g\u00EC trong \u0111o\u1EA1n audio?"
                        oninput="onAnnotationChange(${idx})"
                    ></textarea>
                </div>
                ${outputFields}
            </div>
        `;

        DOM.annotateList.appendChild(item);
    });
}

// Toggle tool call fields based on output type
window.onOutputTypeChange = function (idx) {
    const typeEl = document.getElementById(`output-type-${idx}`);
    const contentGroup = document.getElementById(`output-content-group-${idx}`);
    const toolSection = document.getElementById(`tool-section-${idx}`);
    const type = typeEl.value;

    // Show/hide content field
    if (contentGroup) {
        contentGroup.classList.toggle('hidden', type === 'tool_call');
    }
    // Show/hide tool fields
    if (toolSection) {
        toolSection.classList.toggle('hidden', type === 'text');
    }

    onAnnotationChange(idx);
};

// Collect structured annotation data
window.onAnnotationChange = function (idx) {
    const transcriptEl = document.getElementById(`transcript-${idx}`);
    state.annotations[idx].transcript = transcriptEl ? transcriptEl.value : '';

    const isPhase3 = state.phase === 'phase3_agentic';

    if (isPhase3) {
        const typeEl = document.getElementById(`output-type-${idx}`);
        const contentEl = document.getElementById(`output-content-${idx}`);
        const toolNameEl = document.getElementById(`tool-name-${idx}`);
        const toolArgsEl = document.getElementById(`tool-args-${idx}`);

        const type = typeEl ? typeEl.value : 'text';
        const content = contentEl ? contentEl.value : '';
        const toolName = toolNameEl ? toolNameEl.value.trim() : '';
        const toolArgsRaw = toolArgsEl ? toolArgsEl.value.trim() : '';

        // Parse tool args JSON (gracefully)
        let toolArgs = {};
        if (toolArgsRaw) {
            try { toolArgs = JSON.parse(toolArgsRaw); }
            catch (e) { toolArgs = { _raw: toolArgsRaw }; }
        }

        // Build agnostic output object
        const output = { type };
        if (type === 'text') {
            output.content = content;
        } else if (type === 'tool_call') {
            output.calls = toolName ? [{ name: toolName, args: toolArgs }] : [];
        } else { // mixed
            output.content = content;
            output.calls = toolName ? [{ name: toolName, args: toolArgs }] : [];
        }

        state.annotations[idx].output = output;
    }

    // Mark completed
    const item = document.getElementById(`annotate-${idx}`);
    const hasTranscript = state.annotations[idx].transcript.trim() !== '';
    item.classList.toggle('completed', hasTranscript);

    // Auto-fill augmented samples with same parent
    const sample = state.samples[idx];
    if (!sample.isAugmented) {
        state.samples.forEach((s, i) => {
            if (s.isAugmented && s.parentIdx === idx) {
                autoFillChild(i, idx);
            }
        });
    }

    updateAnnotateProgress();
};

/** Auto-fill an augmented child with its parent's annotation */
function autoFillChild(childIdx, parentIdx) {
    const parent = state.annotations[parentIdx];

    // Transcript
    const childTranscript = document.getElementById(`transcript-${childIdx}`);
    if (childTranscript && !childTranscript.value) {
        childTranscript.value = parent.transcript;
        state.annotations[childIdx].transcript = parent.transcript;
    }

    // Phase 3 structured fields
    if (state.phase === 'phase3_agentic' && parent.output) {
        state.annotations[childIdx].output = JSON.parse(JSON.stringify(parent.output));

        const childType = document.getElementById(`output-type-${childIdx}`);
        const childContent = document.getElementById(`output-content-${childIdx}`);
        const childToolName = document.getElementById(`tool-name-${childIdx}`);
        const childToolArgs = document.getElementById(`tool-args-${childIdx}`);

        if (childType) childType.value = parent.output.type;
        if (childContent && !childContent.value && parent.output.content) {
            childContent.value = parent.output.content;
        }
        if (childToolName && !childToolName.value && parent.output.calls?.[0]) {
            childToolName.value = parent.output.calls[0].name;
        }
        if (childToolArgs && !childToolArgs.value && parent.output.calls?.[0]?.args) {
            const raw = parent.output.calls[0].args._raw;
            childToolArgs.value = raw || JSON.stringify(parent.output.calls[0].args, null, 2);
        }

        // Trigger visibility toggle
        onOutputTypeChange(childIdx);
    }

    const childItem = document.getElementById(`annotate-${childIdx}`);
    childItem?.classList.toggle('completed', childTranscript?.value.trim() !== '');
}

function updateAnnotateProgress() {
    const completed = state.annotations.filter(a => a.transcript.trim() !== '').length;
    DOM.annotateProgressText.textContent = `${completed}/${state.samples.length} đã gán nhãn`;
}

// ============================================
// Step 5: Export — Agnostic Format
// ============================================
//
// Output metadata.jsonl format:
//
// Phase 2 (Alignment):
//   {"audio_file":"wavs/x.wav", "transcript":"...", "augmented_from":null}
//
// Phase 3 (Agentic) — LLM-agnostic:
//   {"audio_file":"wavs/x.wav", "transcript":"...",
//    "output":{"type":"text", "content":"..."}}
//   {"audio_file":"wavs/x.wav", "transcript":"...",
//    "output":{"type":"tool_call", "calls":[{"name":"fn", "args":{}}]}}
//   {"audio_file":"wavs/x.wav", "transcript":"...",
//    "output":{"type":"mixed", "content":"...", "calls":[{"name":"fn", "args":{}}]}}
//
// DataLoader sẽ convert output → LLM-specific format khi train.
//

function buildMetadataEntry(sample, idx) {
    const annotation = state.annotations[idx];
    const entry = {
        audio_file: `wavs/${sample.filename}`,
        transcript: annotation.transcript,
    };

    // Phase 3: structured agnostic output
    if (state.phase === 'phase3_agentic' && annotation.output) {
        // Clean up empty calls
        const output = { ...annotation.output };
        if (output.calls && output.calls.length === 0) delete output.calls;
        if (output.content === '') delete output.content;
        entry.output = output;
    }

    // Augmentation metadata
    if (sample.isAugmented) {
        entry.augmented_from = `wavs/${state.samples[sample.parentIdx].filename}`;
        entry.aug_type = sample.augType;
    } else {
        entry.augmented_from = null;
    }

    return entry;
}

function goToExport() {
    DOM.exportSection.classList.remove('hidden');
    DOM.exportSection.scrollIntoView({ behavior: 'smooth' });

    const lines = state.samples.map((sample, idx) =>
        JSON.stringify(buildMetadataEntry(sample, idx), null, 0)
    );

    DOM.exportPreview.innerHTML = `
        <div class="file-count">\uD83D\uDCC1 ${state.samples.length} files \u2014 ${state.phase} (agnostic format)</div>
        <pre>${lines.join('\n')}</pre>
    `;
}

async function exportDataset() {
    DOM.exportBtn.disabled = true;
    DOM.exportBtn.textContent = '\u23F3 \u0110ang \u0111\u00F3ng g\u00F3i...';

    try {
        const zip = new JSZip();
        const folder = zip.folder(state.phase);
        const wavsFolder = folder.folder('wavs');

        // Add audio files
        for (const sample of state.samples) {
            let blob = sample.blob;

            // If original (webm), convert to WAV
            if (!sample.isAugmented && sample.blob.type !== 'audio/wav') {
                try {
                    const audioBuffer = await decodeAudioBlob(sample.blob);
                    blob = audioBufferToWavBlob(audioBuffer);
                } catch (e) {
                    console.warn(`Failed to convert ${sample.filename} to WAV, using original format`, e);
                }
            }

            wavsFolder.file(sample.filename, blob);
        }

        // Build metadata.jsonl — agnostic format
        const metadataLines = state.samples.map((sample, idx) =>
            JSON.stringify(buildMetadataEntry(sample, idx))
        );

        folder.file('metadata.jsonl', metadataLines.join('\n'));

        // Generate and download
        const content = await zip.generateAsync({ type: 'blob' });
        const url = URL.createObjectURL(content);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${state.phase}_dataset.zip`;
        a.click();
        URL.revokeObjectURL(url);

        DOM.exportBtn.textContent = '\u2705 \u0110\u00E3 t\u1EA3i xu\u1ED1ng!';
    } catch (err) {
        console.error('Export failed:', err);
        DOM.exportBtn.textContent = '\u274C L\u1ED7i export';
    }

    setTimeout(() => {
        DOM.exportBtn.disabled = false;
        DOM.exportBtn.innerHTML = '<span class="btn-icon">\uD83D\uDCE6</span> Download ZIP';
    }, 3000);
}

// ============================================
// Make reRecord accessible globally
// ============================================
window.reRecord = reRecord;

// ============================================
// Boot
// ============================================
document.addEventListener('DOMContentLoaded', init);
