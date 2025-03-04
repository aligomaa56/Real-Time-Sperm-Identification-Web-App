document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const previewImg = document.getElementById('previewImg');
    const connectionStatus = document.getElementById('connectionStatus');
    const captureStatus = document.getElementById('captureStatus');
    
    // Stats elements
    const liveCount = document.getElementById('liveCount');
    const deadCount = document.getElementById('deadCount');
    const abnormalCount = document.getElementById('abnormalCount');
    const totalCount = document.getElementById('totalCount');
    const livePercentage = document.getElementById('livePercentage');
    const deadPercentage = document.getElementById('deadPercentage');
    const abnormalPercentage = document.getElementById('abnormalPercentage');
    
    // Socket connection
    const socket = io();
    
    // Variables
    let mediaStream = null;
    let captureInterval = null;
    const captureRate = 1000; // 1 second between captures
    
    // Connect to socket
    socket.on('connect', () => {
        connectionStatus.textContent = 'Status: Connected';
        connectionStatus.style.color = '#2ecc71';
        startBtn.disabled = false;
    });
    
    socket.on('disconnect', () => {
        connectionStatus.textContent = 'Status: Disconnected';
        connectionStatus.style.color = '#e74c3c';
        stopCapture();
    });
    
    // Handle detection results
    socket.on('detection_results', (data) => {
        // Update the preview image
        previewImg.src = data.image;
        previewImg.style.display = 'block';
        
        // Update stats
        liveCount.textContent = data.stats.live;
        deadCount.textContent = data.stats.dead;
        abnormalCount.textContent = data.stats.abnormal;
        totalCount.textContent = data.stats.total;
        
        livePercentage.textContent = `${data.stats.live_percentage.toFixed(1)}%`;
        deadPercentage.textContent = `${data.stats.dead_percentage.toFixed(1)}%`;
        abnormalPercentage.textContent = `${data.stats.abnormal_percentage.toFixed(1)}%`;
    });
    
    // Handle errors
    socket.on('error', (data) => {
        console.error('Server error:', data.message);
        alert(`Error: ${data.message}`);
    });
    
    // Start screen capture
    startBtn.addEventListener('click', async () => {
        try {
            // Request screen capture
            mediaStream = await navigator.mediaDevices.getDisplayMedia({
                video: {
                    cursor: "always"
                },
                audio: false
            });
            
            // Create video element to capture frames
            const video = document.createElement('video');
            video.srcObject = mediaStream;
            video.onloadedmetadata = () => {
                video.play();
                
                // Start capture interval
                captureInterval = setInterval(() => {
                    captureFrame(video);
                }, captureRate);
            };
            
            // Update UI
            startBtn.disabled = true;
            stopBtn.disabled = false;
            captureStatus.textContent = 'Capture: Active';
            captureStatus.style.color = '#2ecc71';
            
            // Handle stream ending (user stops sharing)
            mediaStream.getVideoTracks()[0].onended = () => {
                stopCapture();
            };
            
        } catch (error) {
            console.error('Error starting capture:', error);
            alert('Failed to start screen capture. Please ensure you grant the necessary permissions.');
        }
    });
    
    // Stop screen capture
    stopBtn.addEventListener('click', () => {
        stopCapture();
    });
    
    // Capture frame and send to server
    function captureFrame(video) {
        // Check if we're online
        if (!navigator.onLine) {
            alert('Internet connection lost. Screen capture stopped.');
            stopCapture();
            return;
        }
        
        // Create canvas to draw the frame
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // Convert to base64 and send to server
        const imageData = canvas.toDataURL('image/jpeg', 0.7);
        socket.emit('screen_capture', { image: imageData });
    }
    
    // Stop capture
    function stopCapture() {
        // Clear interval
        if (captureInterval) {
            clearInterval(captureInterval);
            captureInterval = null;
        }
        
        // Stop media stream
        if (mediaStream) {
            mediaStream.getTracks().forEach(track => track.stop());
            mediaStream = null;
        }
        
        // Update UI
        startBtn.disabled = false;
        stopBtn.disabled = true;
        captureStatus.textContent = 'Capture: Inactive';
        captureStatus.style.color = '#e74c3c';
    }
    
    // Check for internet connection
    window.addEventListener('offline', () => {
        alert('Internet connection lost. Screen capture stopped.');
        stopCapture();
    });
});