let mediaRecorder;
let audioChunks = [];
let audioBlob = null;
let audioUrl = null;
let stream = null;

// --- EARLY CHECK: Verify getUserMedia Support ---
// (Keep this check as it was)
if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    // ... (keep the existing error handling for missing API)
    console.error("getUserMedia not supported!");
    // ... (disable buttons, show alert etc.)
}

// Function to setup recording for a specific form
// Parameter 'needsUserData' now indicates if username/language are needed
function setupRecording(formId, recordBtnId, stopBtnId, submitBtnId, playbackId, statusId, messageId, loadingId, endpointUrl, needsUserData) {
    const form = document.getElementById(formId);
    const recordButton = document.getElementById(recordBtnId);
    const stopButton = document.getElementById(stopBtnId);
    const submitButton = document.getElementById(submitBtnId);
    const audioPlayback = document.getElementById(playbackId);
    const recordingStatus = document.getElementById(statusId);
    const messageArea = document.getElementById(messageId);
    const loadingIndicator = document.getElementById(loadingId);
    const usernameInput = document.getElementById('username'); // Relevant for enrollment
    const languageSelect = document.getElementById('language'); // Get language select element

    // --- Reset state function ---
    function resetUI() {
        // ...(Keep existing reset logic)...
        recordButton.disabled = false;
        stopButton.disabled = true;
        submitButton.disabled = true;
        recordingStatus.textContent = '';
        audioPlayback.style.display = 'none';
        audioPlayback.src = '';
        if (audioUrl) { URL.revokeObjectURL(audioUrl); audioUrl = null; }
        audioBlob = null;
        audioChunks = [];
        loadingIndicator.style.display = 'none';
        if (stream) { stream.getTracks().forEach(track => track.stop()); stream = null; }
    }

    resetUI(); // Initial reset

    // --- Event Listeners ---
    recordButton.addEventListener('click', async () => {
        // ...(Keep existing record button logic)...
         resetUI();
         messageArea.textContent = '';
         messageArea.className = 'message-area';
         try {
            stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            recordButton.disabled = true;
            stopButton.disabled = false;
            recordingStatus.textContent = '🔴 Recording...';
            const options = { mimeType: 'audio/wav' }; // Prefer WAV
             try {
                 if (!MediaRecorder.isTypeSupported('audio/wav')) { options.mimeType = 'audio/webm'; }
                 mediaRecorder = new MediaRecorder(stream, options);
             } catch (e) { throw new Error("No suitable audio recording format supported."); }

            mediaRecorder.ondataavailable = event => { if (event.data.size > 0) { audioChunks.push(event.data); } };
            mediaRecorder.onstop = () => {
                console.log("Recording stopped.");
                audioBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType });
                audioUrl = URL.createObjectURL(audioBlob);
                audioPlayback.src = audioUrl;
                audioPlayback.style.display = 'block';
                recordingStatus.textContent = '⏹️ Recording stopped. Ready to submit.';
                submitButton.disabled = false;
                audioChunks = [];
                if (stream) { stream.getTracks().forEach(track => track.stop()); stream = null; }
                recordButton.disabled = false;
                stopButton.disabled = true;
            };
             mediaRecorder.onerror = (event) => { console.error(`MediaRecorder error: ${event.error}`); resetUI(); };
            mediaRecorder.start();
         } catch (error) {
             // ...(Keep existing detailed error handling for getUserMedia)...
             console.error("Error accessing microphone:", error);
             let userMessage = "Could not access microphone. ";
              // ... (switch statement for error.name) ...
             recordingStatus.textContent = userMessage;
             alert("Microphone access failed: " + error.message);
             resetUI();
         }
    });

    stopButton.addEventListener('click', () => {
         // ...(Keep existing stop button logic)...
         if (mediaRecorder && mediaRecorder.state === 'recording') { mediaRecorder.stop(); }
         else { console.warn("Stop clicked but not recording."); if (stream) {stream.getTracks().forEach(track => track.stop()); stream = null;} resetUI(); }
    });

    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        messageArea.textContent = '';
        messageArea.className = 'message-area';

        if (!audioBlob) {
            messageArea.textContent = 'Please record audio first.';
            messageArea.className = 'message-area message-error';
            return;
        }

        // --- Get Language Value ---
        const selectedLanguage = languageSelect ? languageSelect.value : null;
        if (!selectedLanguage) {
            messageArea.textContent = 'Please select a language.';
            messageArea.className = 'message-area message-error';
            return;
        }
        // ---

        loadingIndicator.style.display = 'block';
        submitButton.disabled = true;
        recordButton.disabled = true;
        stopButton.disabled = true;

        const formData = new FormData();
        const fileExtension = (mediaRecorder?.mimeType?.includes('wav')) ? 'wav' : 'webm';
        const filename = `recording.${fileExtension}`;
        formData.append('audio_data', audioBlob, filename);

        // --- Append User Data (Username and Language) conditionally ---
        formData.append('language', selectedLanguage); // Always send language

        if (needsUserData) { // Only add username if required (enrollment)
            const username = usernameInput.value.trim();
            if (!username) {
                 messageArea.textContent = 'Username is required for enrollment.';
                 messageArea.className = 'message-area message-error';
                 loadingIndicator.style.display = 'none';
                 submitButton.disabled = false; // Re-enable
                 recordButton.disabled = false;
                return; // Stop submission
            }
            formData.append('username', username);
            console.log(`Submitting Enrollment: User=${username}, Lang=${selectedLanguage}, Audio=${filename}`);
        } else {
            console.log(`Submitting Login: Lang=${selectedLanguage}, Audio=${filename}`);
        }
        // ---

        try {
            const response = await fetch(endpointUrl, { method: 'POST', body: formData });
            const result = await response.json();

            if (response.ok && result.status === 'success') {
                messageArea.textContent = result.message;
                messageArea.className = 'message-area message-success';
                if (!needsUserData) { // Successful login
                    recordingStatus.textContent = '✅ Verification Successful! Redirecting...';
                     setTimeout(() => { window.location.href = '/dashboard'; }, 1500);
                } else { // Successful enrollment
                     recordingStatus.textContent = '✅ Enrollment Successful!';
                     form.reset(); // Clears username and resets language dropdown
                     resetUI(); // Reset recording controls and audio
                     // Manually trigger prompt update after form reset
                     if(typeof updateEnrollmentPrompts === "function") updateEnrollmentPrompts();
                }
            } else { // Handle errors
                messageArea.textContent = `Error: ${result.message || `Server responded with status ${response.status}`}`;
                messageArea.className = 'message-area message-error';
                recordingStatus.textContent = '❌ Operation Failed.';
                 submitButton.disabled = false;
                 recordButton.disabled = false;
            }

        } catch (error) {
            // ...(Keep existing network error handling)...
             console.error('Error submitting form:', error);
             messageArea.textContent = 'Network error or server issue.';
             messageArea.className = 'message-area message-error';
             recordingStatus.textContent = '❌ Network Error.';
             submitButton.disabled = false;
             recordButton.disabled = false;
        } finally {
             loadingIndicator.style.display = 'none';
             // Buttons re-enabled on failure/enroll-success cases above.
        }
    });
}
