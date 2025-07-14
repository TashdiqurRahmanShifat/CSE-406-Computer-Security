function app() {
    return {
        latencyResults: null,  // To store latency results
        isCollecting: false,  // To disable the button during collection
        showingTraces: false,  // To toggle UI between showing latency or traces
        traces: [],  // To store trace heatmaps
        statusMessage: '',  // Status message during collection
        message: '',  // General message display


        // For real-time detection
        isRealTimeActive: false,
        realTimeInterval: null,
        currentPrediction: {
            website: 'Unknown',
            confidence: 0
        },
        realTimeStatus: 'Idle',
        lastUpdateTime: 'Never',

        // Function to collect latency data
        async collectLatencyData() {
            this.isCollecting = true;  // Disable the button
            this.latencyResults = null;  // Clear previous results
            this.showingTraces = false;
            this.statusMessage = 'Collecting latency data...';

            try {
                console.log("Starting latency data collection...");
                const worker = new Worker("warmup.js");  // Create a worker
                console.log("Worker created, setting up message handlers...");

                // Set up promise to handle worker completion
                const workerPromise = new Promise((resolve, reject) => {
                  console.log("Setting up worker message handlers...");
                    worker.onmessage = (e) => {
                        console.log("Received results from worker:", e.data);
                        this.latencyResults = e.data;  // Store results
                        worker.terminate(); // Clean up the worker
                        console.log("Worker terminated after receiving results.");
                        resolve(e.data);
                    };

                    worker.onerror = (err) => {
                        console.error("Worker error:", err);
                        worker.terminate();
                        reject(err);
                    };
                });

                console.log("Sending start message to worker...");
                worker.postMessage("start");  // Start the worker task
                console.log("Waiting for worker to complete...");
                await workerPromise; // Wait for completion
                console.log("Latency data collection complete!");

            } catch (error) {
                console.error("Error collecting latency data:", error);
                alert("Error collecting latency data. Check console for details.");
            } finally {
                this.isCollecting = false;  // Re-enable the button
                this.statusMessage = '';
            }
        },

        // Function to collect trace data
        async collectTraceData() {
            this.isCollecting = true;
            this.showingTraces = false;//This hides the trace display section while new data is being collected
            this.statusMessage = 'Collecting trace data... This will take 10 seconds.';

            try {
                console.log("Starting trace data collection...");
                const worker = new Worker("worker.js");

                // Set up promise to handle worker completion
                const workerPromise = new Promise((resolve, reject) => {
                    worker.onmessage = (e) => {
                        console.log("Received trace results from worker:", e.data);
                        worker.terminate();
                        resolve(e.data);
                    };

                    worker.onerror = (err) => {
                        console.error("Worker error:", err);
                        worker.terminate();
                        reject(err);
                    };
                });

                worker.postMessage("start");
                //Receive trace data from the worker by front-end
                const traceData = await workerPromise;

                if (traceData && traceData.length > 0) {
                    this.statusMessage = 'Sending trace data to backend for visualization...';
                    
                    // Send trace data to backend for visualization
                    const response = await fetch('/collect_trace', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            trace_data: traceData,
                            timestamp: new Date().toISOString()
                        })
                    });

                    if (response.ok) {
                        const result = await response.json();
                        
                        // Add the new trace to our traces array
                        this.traces.push({
                            heatmap_url: result.heatmap_url,
                            info: result.info || `Min: ${Math.min(...traceData)}, Max: ${Math.max(...traceData)}, Range: ${Math.max(...traceData) - Math.min(...traceData)}, Samples: ${traceData.length}`
                        });
                        
                        this.showingTraces = true;
                        this.message = 'Trace data collected and visualized successfully!';
                        
                        // Clear message after 3 seconds
                        setTimeout(() => {
                            this.message = '';
                        }, 3000);
                    } else {
                        throw new Error('Failed to process trace data on backend');
                    }
                } else {
                    throw new Error('No trace data received from worker');
                }

            } catch (error) {
                console.error("Error collecting trace data:", error);
                this.message = `Error collecting trace data: ${error.message}`;
            } finally {
                this.isCollecting = false;
                this.statusMessage = '';
            }
        },

        // Function to download traces
        async downloadTraces() {
            try {
                const response = await fetch('/download_traces');
                if (response.ok) {
                    const blob = await response.blob(); //  blob (binary large object) suitable for file download
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = 'traces.json';
                    document.body.appendChild(a);
                    a.click();
                    window.URL.revokeObjectURL(url);
                    document.body.removeChild(a);
                    this.message = 'Traces downloaded successfully!';
                } else {
                    throw new Error('Failed to download traces');
                }
            } catch (error) {
                console.error("Error downloading traces:", error);
                this.message = `Error downloading traces: ${error.message}`;
            }
        },

        // Function to refresh results
        refreshResults() {
            // This will reload the page to refresh all results
            window.location.reload();
        },

        // Function to clear all results
        async clearResults() {
            try {
                const response = await fetch('/api/clear_results', {
                    method: 'POST'
                });
                if (response.ok) {
                    this.traces = [];
                    this.latencyResults = null;
                    this.showingTraces = false;
                    this.message = 'All results cleared successfully!';
                    
                    // Clear the message after 2 seconds
                    setTimeout(() => {
                        this.message = '';
                    }, 2000);
                } else {
                    throw new Error('Failed to clear results');
                }
            } catch (error) {
                console.error("Error clearing results:", error);
                this.message = `Error clearing results: ${error.message}`;
            }
        },

        

        // Toggle real-time detection - now does single prediction
        async toggleRealTimeDetection() {
            if (this.isRealTimeActive) {
                // If currently predicting, do nothing (let it finish)
                return;
            } else {
                // Perform single prediction
                await this.performSinglePrediction();
            }
        },

        // Perform single prediction (no continuous loop)
        async performSinglePrediction() {
            this.isRealTimeActive = true;
            this.realTimeStatus = 'Starting...';
            
            try {
                // Run prediction once
                await this.performRealTimePrediction();
            } finally {
                // Always reset state after prediction
                this.isRealTimeActive = false;
            }
        },

        // // Stop real-time detection
        // stopRealTimeDetection() {
        //     this.isRealTimeActive = false;
        //     this.realTimeStatus = 'Stopped';
            
        //     if (this.realTimeInterval) {
        //         clearInterval(this.realTimeInterval);
        //         this.realTimeInterval = null;
        //     }
        // },

// Perform a single real-time prediction
async performRealTimePrediction() {
    try {
        this.realTimeStatus = 'Collecting trace...';
        
        // Collect trace data using existing worker
        const worker = new Worker("worker.js");
        
        const workerPromise = new Promise((resolve, reject) => {
            worker.onmessage = (e) => {
                worker.terminate();
                resolve(e.data);
            };
            worker.onerror = (err) => {
                worker.terminate();
                reject(err);
            };
        });
        
        worker.postMessage("start");
        const traceData = await workerPromise;
        
        if (traceData && traceData.length > 0) {
            this.realTimeStatus = 'Predicting...';
            
            // Send trace data to prediction endpoint
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    trace_data: traceData
                })
            });
            
            if (response.ok) {
                const result = await response.json();
                this.currentPrediction = {
                    website: result.predicted_website,
                    confidence: result.confidence
                };
                this.realTimeStatus = 'Active';
                this.lastUpdateTime = new Date().toLocaleTimeString();
            } else {
                throw new Error('Prediction failed');
            }
        }
        
    } catch (error) {
        console.error("Real-time prediction error:", error);
        this.realTimeStatus = 'Error';
        this.currentPrediction = {
            website: 'Error',
            confidence: 0
        };
    }
}

    };
}