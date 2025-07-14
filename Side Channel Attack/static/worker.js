/* Find the cache line size by running `getconf -a | grep CACHE` */
const LINESIZE = 64;
/* Find the L3 size by running `getconf -a | grep CACHE` - Updated with your values */
const LLCSIZE = 8388608; // 8MB from L3 cache size
/* Collect traces for 10 seconds; you can vary this */
const TIME = 10000;
/* Collect traces every 10ms; you can vary this */
const P = 10; 


//Each website creates a unique pattern in the sweep counts array, which becomes its fingerprint
function sweep(P) {
    /*
     * Implement this function to run a sweep of the cache.
     * 1. Allocate a buffer of size LLCSIZE.
     * 2. Read each cache line (read the buffer in steps of LINESIZE).
     * 3. Count the number of times each cache line is read in a time period of P milliseconds.
     * 4. Store the count in an array of size K, where K = TIME / P.
     * 5. Return the array of counts.
     */
    
    try {
        // Calculate number of measurements
        const K = Math.floor(TIME / P);
        const sweepCounts = [];
        
        // 1. Allocate a buffer of size LLCSIZE
        const buffer = new ArrayBuffer(LLCSIZE);
        const view = new Uint8Array(buffer);
        // const numCacheLines = Math.floor(LLCSIZE / LINESIZE);
        
        // 2. Perform K measurements, each for P milliseconds
        for (let measurement = 0; measurement < K; measurement++) {
            const startTime = performance.now();
            let sweepCount = 0;
            
            // Keep sweeping until P milliseconds have passed
            while ((performance.now() - startTime) < P) {
                // Read through buffer at intervals of LINESIZE to access different cache lines
                for (let i = 0; i < LLCSIZE; i += LINESIZE) {
                    // Access the memory location to trigger cache activity
                    const dummy = view[i];
                }
                sweepCount++;
            }
            //A sweep refers to one complete pass through a memory buffer, touching (reading) each cache line once.
            sweepCounts.push(sweepCount);
        }
        
        return sweepCounts;
        
    } catch (error) {
        console.error('Error in sweep function:', error);
        return null;
    }
}   

self.addEventListener('message', function(e) {
    if (e.data === 'start') {
        const result = sweep(P);
        self.postMessage(result);
    }
});
