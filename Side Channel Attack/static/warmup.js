const LINESIZE = 64; // Cache line size in bytes

function readNlines(n) {
    /*
     * Implement this function to read n cache lines.
     * 1. Allocate a buffer of size n * LINESIZE.
     * 2. Read each cache line (read the buffer in steps of LINESIZE) 10 times.
     * 3. Collect total time taken in an array using `performance.now()`.
     * 4. Return the median of the time taken in milliseconds.
     */
    try 
    {
        // 1. Allocate a buffer of size n * LINESIZE using ArrayBuffer for actual memory allocation
        const bufferSize = n * LINESIZE;
        const buffer = new ArrayBuffer(bufferSize);// creates consecutive bytes in memory
        const view = new Uint8Array(buffer); // Create typed array view that gives you direct access to the raw bytes,that is exact byte-level addressing
        
        // Initialize the buffer to ensure it's allocated in memory
        for (let i = 0; i < bufferSize; i += LINESIZE) {
            view[i] = Math.floor(Math.random() * 256); // Only writes to the first byte of each cache line.Withou this,there is a chance that modern browsers optimize memory access
        }
        
        const times = [];

        
        // Loop to perform the read operation 10 times for accuracy
        for (let iteration = 0; iteration < 10; iteration++) {
            const startTime = performance.now();
            
            // Read through the buffer at intervals of LINESIZE to access different cache lines
            for (let j = 0; j < n; j++) {
                const offset = j * LINESIZE;
                // Access the memory location to trigger cache activity
                const dummy = view[offset];
            }
            const endTime = performance.now();
            times.push(endTime - startTime);
        }
        
        
        // Sort the times array to find the median
        times.sort((a, b) => a - b);
        return times[Math.floor(times.length / 2)]; // Return the median time
        
    } 
    catch (error) {
        console.error(`Error in readNlines for n=${n}:`, error);
        return null;
    }
}

self.addEventListener("message", function (e) {
    console.log("Worker received message:", e.data);
    if (e.data === "start") {

        // Object to store results for each n
        const results = {};
        
        // Linear progression: multiply by 10 each iteration
        for (let n = 1; n <= 10000000; n *= 10) {
            console.log(`Starting read for n = ${n}`);
            
            try {
                const result = readNlines(n);
                if (result !== null) {
                    results[n] = result;
                    console.log(`n=${n}: ${result.toFixed(2)}ms`);
                } else {
                    console.log(`Failed to read ${n} cache lines, stopping here`);
                    break;
                }
            } catch (error) {
                console.error(`Failed at n = ${n}:`, error);
                break;
            }
        }
        
        console.log("Sending results back to the main thread:", results);
        self.postMessage(results);
    }
});
