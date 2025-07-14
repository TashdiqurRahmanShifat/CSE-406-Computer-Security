import time
import json
import os
import signal
import sys
import random
import traceback
import socket
import subprocess
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
import database
from database import Database
# Add these constants after the existing imports
import os

# Get current working directory and create HEATMAPS folder
HEATMAPS_DIR = os.path.join(os.getcwd(), "HEATMAPS")
os.makedirs(HEATMAPS_DIR, exist_ok=True)

WEBSITES = [
    # websites of your choice
    "https://cse.buet.ac.bd/moodle/",
    "https://google.com",
    "https://prothomalo.com",
]

TRACES_PER_SITE = 1100  # Number of traces to collect per website
FINGERPRINTING_URL = "http://localhost:5000" 
OUTPUT_PATH = "dataset.json"

# Initialize the database to save trace data reliably
database.db = Database(WEBSITES)

""" Signal handler to ensure data is saved before quitting. """
def signal_handler(sig, frame):
    print("\nReceived termination signal. Exiting gracefully...")
    try:
        database.db.export_to_json(OUTPUT_PATH)
    except:
        pass
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)


"""
Some helper functions to make your life easier.
"""

def is_server_running(host='127.0.0.1', port=5000):
    """Check if the Flask server is running."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex((host, port)) # 0 means success, anything else means failure
    sock.close()
    return result == 0


def setup_webdriver():
    """Enhanced webdriver setup"""
    chrome_options = Options()
    chrome_options.add_argument("--window-size=1920,1080") # Sets a consistent window size
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage") # prevents shared memory issues in containerized environments
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-features=IsolateOrigins,site-per-process")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    service = Service(ChromeDriverManager().install())#automatically downloads and installs the correct version of ChromeDriver that matches your installed Chrome browser
    driver = webdriver.Chrome(service=service, options=chrome_options)# creates a service object that manages the ChromeDriver process
    #This eliminates the need to manually download and manage ChromeDriver executables
    
    # Mask automation
    driver.execute_script("""
        Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
        window.chrome = { runtime: {} };
    """)
    
    return driver


def clear_trace_results(driver, wait):
    """Clear all results from the backend via API call."""
    try:
        result = driver.execute_script("""
            return fetch('/api/clear_results', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'}
            })
            .then(response => response.json())
            .then(data => {
                console.log('Clear results response:', data);
                return data.success;
            })
            .catch(error => {
                console.error('Clear results error:', error);
                return false;
            });
        """)
        
        if result:
            print("  - Cleared previous results via API")
        else:
            print("  - Warning: API clear request failed")
        
        time.sleep(1)  # Give time for any UI updates
        
    except Exception as e:
        print(f"  - Warning: Could not clear results: {e}")
 
# If the script crashes and restarts, it doesn't start from zero - it continues where it left off.  
def is_collection_complete():
    """Check if target number of traces have been collected."""
    current_counts = database.db.get_traces_collected()
    remaining_counts = {website: max(0, TRACES_PER_SITE - count) 
                      for website, count in current_counts.items()}
    return sum(remaining_counts.values()) == 0

def start_flask_server():
    """Start the Flask server if it's not running."""
    if not is_server_running():
        print("Starting Flask server...")
        try:
            # Start Flask server in background
            subprocess.Popen([sys.executable, "app.py"], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
            
            # Wait for server to start
            for _ in range(10):
                time.sleep(1)
                #After each second, it checks if the server is running.
                if is_server_running():
                    print("Flask server started successfully")
                    return True
            
            print("Warning: Flask server may not have started properly")
            return False
        except Exception as e:
            print(f"Error starting Flask server: {e}")
            return False
    else:
        print("Flask server is already running")
        return True


def collect_single_trace(driver, wait, website_url, trace_number):
    """Corrected trace collection with proper sequence"""
    try:
        print(f"  - Collecting trace for {website_url}")
        
        # code waits for web elements to be ready, making browser automation stable and robust.
        # wait for 20 seconds before starting the trace collection
        wait = WebDriverWait(driver, 20)
        fingerprinting_tab = driver.current_window_handle
        
        # Ensure we're on the fingerprinting page
        if driver.current_url != FINGERPRINTING_URL:
            driver.get(FINGERPRINTING_URL)
            time.sleep(3)
        
        # Click the collect trace button to start monitoring
        collect_button = wait.until(EC.element_to_be_clickable(
            (By.XPATH, "//button[contains(text(), 'Collect Trace')]")))
        driver.execute_script("arguments[0].click();", collect_button)
        print("    Started trace collection monitoring...")
        time.sleep(2)  # Give the monitoring time to initialize
        
        # Open target website in new tab
        driver.execute_script("window.open('');")
        # Switches Selenium’s context to the new tab (so future commands apply to the new website).
        driver.switch_to.window(driver.window_handles[-1])
        
        try:
            # Load and interact with the target website
            driver.get(website_url)
            # Waits until the <body> element is present,that is,the page has loaded
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            # Waits a bit more to be sure the site is fully ready.
            time.sleep(3)
            
            # Simulate user activity on target website
            print("    Simulating user activity...")
            for _ in range(random.randint(3, 7)):
                scroll_amount = random.randint(200, 800)
                driver.execute_script(f"window.scrollBy(0, {scroll_amount});")
                time.sleep(random.uniform(0.5, 1.5))
            
            # Scroll back up
            for _ in range(random.randint(2, 4)):
                scroll_amount = random.randint(-800, -200)
                driver.execute_script(f"window.scrollBy(0, {scroll_amount});")
                time.sleep(random.uniform(0.5, 1.0))
            
            # Try to click on some elements
            try:
                clickable_elements = driver.find_elements(By.TAG_NAME, "a")[:3]
                for element in clickable_elements:
                    if element.is_displayed() and element.is_enabled():
                        try:
                            ActionChains(driver).move_to_element(element).perform()
                            time.sleep(random.uniform(0.2, 0.5))
                        except:
                            pass
            except:
                pass
                
        except Exception as e:
            print(f"    Warning: Could not fully load/interact with {website_url}: {e}")
        
        # Switch back to fingerprinting tab and close target tab
        driver.switch_to.window(fingerprinting_tab)
        
        # Close the target website tab
        if len(driver.window_handles) > 1:
            for handle in driver.window_handles:
                #If there’s more than one tab open, closes all tabs except the fingerprinting tab.
                if handle != fingerprinting_tab:
                    driver.switch_to.window(handle)
                    driver.close()
            # Ensures you’re back on the correct tab for the next operation.
            driver.switch_to.window(fingerprinting_tab)
        
        # Wait for trace collection to complete
        print("    Waiting for trace collection to finish...")
        success = False
        
        for attempt in range(4):
            try:
                # Check for heatmap or trace completion
                heatmap = wait.until(EC.presence_of_element_located(
                    # Waits for the heatmap/trace result to appear
                    (By.CSS_SELECTOR, ".heatmap-container img, canvas.heatmap")))
                # the trace was collected successfully!
                success = True
                break
            except:
                time.sleep(3)
        
        if success:
            print("    Trace collection completed successfully")
            save_heatmap(driver, website_url, trace_number)
            return True
        else:
            print("    Trace collection failed - no data found")
            return False
     
    except Exception as e:
        print(f"    Error in trace collection: {e}")
        traceback.print_exc()
        
        # Cleanup: ensure we're back on fingerprinting tab
        try:
            driver.switch_to.window(fingerprinting_tab)
            # Close any extra tabs
            for handle in driver.window_handles:
                if handle != fingerprinting_tab:
                    driver.switch_to.window(handle)
                    driver.close()
            driver.switch_to.window(fingerprinting_tab)
        except:
            pass
        return False


#collecting and saving browser fingerprinting data from various websites
def collect_fingerprints(driver, target_counts=None):
    """ Implement the main logic to collect fingerprints.
    1. Calculate the number of traces remaining for each website
    2. Open the fingerprinting website
    3. Collect traces for each website until the target number is reached
    4. Save the traces to the database
    5. Return the total number of new traces collected
    """
    
    wait = WebDriverWait(driver, 10)
    total_collected = 0
    
    try:
        # Calculate the number of traces remaining for each website
        # The function first retrieves the current number of traces already collected from the database
        current_counts = database.db.get_traces_collected()
        # If no target trace count (target_counts) is provided, a default target value (TRACES_PER_SITE) is used for each website in WEBSITES
        if target_counts is None:
            target_counts = {website: TRACES_PER_SITE for website in WEBSITES}
        
        remaining_counts = {website: max(0, target_counts[website] - current_counts.get(website, 0)) 
                          for website in WEBSITES}
        
        print(f"Current trace counts: {current_counts}")
        print(f"Target counts: {target_counts}")
        print(f"Remaining to collect: {remaining_counts}")
        
        # Open the fingerprinting website
        driver.get(FINGERPRINTING_URL)
        time.sleep(2)
        
        # Clear any previous results
        clear_trace_results(driver, wait)
        
        # Collect traces for each website until the target number is reached
        for website_url in WEBSITES:
            remaining = remaining_counts[website_url]
            if remaining <= 0:
                print(f"Skipping {website_url} - target already reached")
                continue
                
            print(f"\nCollecting {remaining} traces for {website_url}")
            
            for i in range(remaining):
                print(f"  Collecting trace {i+1}/{remaining} for {website_url}")
                
                # Collect a single trace
                success = collect_single_trace(driver, wait, website_url,i+1)
                
                if success:
                    # Save the trace to the database
                    # Get the trace data from the backend
                    trace_data = driver.execute_script("""
                        // Get the last collected trace data
                        return fetch('/download_traces')
                            .then(response => response.json())
                            .then(data => {
                                if (data && data.length > 0) {
                                    return data[data.length - 1].trace_data;
                                }
                                return null;
                            })
                            .catch(() => null);
                    """)
                    
                    if trace_data:
                        # Save to database
                        site_idx = WEBSITES.index(website_url)
                        # The save_trace function is responsible for storing the trace data in the database
                        if database.db.save_trace(website_url, site_idx, trace_data):
                            # total_collected keeps track of the number of new traces collected during the session
                            total_collected += 1
                            print(f"    Saved trace {i+1} to database")
                        else:
                            print(f"    Failed to save trace {i+1} to database")
                    else:
                        print(f"    Warning: Could not retrieve trace data for saving")
                    
                    # Clear results for next collection
                    clear_trace_results(driver, wait)

                    
                else:
                    print(f"    Failed to collect trace {i+1}")
                
                # Small delay between collections
                time.sleep(random.uniform(1, 3))
                
        print(f"\nCollection session complete. Total new traces collected: {total_collected}")
        return total_collected
        
    except Exception as e:
        print(f"Error in collect_fingerprints: {e}")
        traceback.print_exc()
        return total_collected
    

def save_heatmap(driver, website_url, trace_number):
    """Save the heatmap image with better element detection and error handling"""
    try:
        # Wait for the heatmap to be present
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".heatmap-container canvas, .heatmap-container img"))
        )
        
        # Try multiple methods to get the heatmap data
        img_base64 = driver.execute_script("""
            // Try canvas first
            let canvas = document.querySelector('.heatmap-container canvas');
            if (canvas) {
                return canvas.toDataURL('image/png').split(',')[1];
            }
            
            // Try img element next
            let img = document.querySelector('.heatmap-container img');
            if (img && img.src.startsWith('data:image')) {
                return img.src.split(',')[1];
            }
            
            // Try other canvas
            canvas = document.querySelector('canvas.heatmap');
            if (canvas) {
                return canvas.toDataURL('image/png').split(',')[1];
            }
            
            return null;
        """)
        
        if not img_base64:
            print("    No heatmap data found")
            return False
            
        # Create filename with timestamp to avoid overwrites
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        website_name = website_url.replace('https://', '').replace('http://', '').split('/')[0]
        filename = f"{website_name}_trace_{trace_number}_{timestamp}.png"
        filepath = os.path.join(HEATMAPS_DIR, filename)
        
        # Save the image
        import base64
        with open(filepath, 'wb') as f:
            f.write(base64.b64decode(img_base64))
        print(f"    ✓ Saved heatmap to {filepath}")
        return True
        
    except Exception as e:
        print(f"    Error saving heatmap: {e}")
        traceback.print_exc()
        return False

def main():
    """ Implement the main function to start the collection process.
    1. Check if the Flask server is running
    2. Initialize the database
    3. Set up the WebDriver
    4. Start the collection process, continuing until the target number of traces is reached
    5. Handle any exceptions and ensure the WebDriver is closed at the end
    6. Export the collected data to a JSON file
    7. Retry if the collection is not complete
    """
    # driver variable that will be used to interact with the web browser
    driver = None
    # maximum number of retries for the collection process in case of failures
    max_retries = 3
    # keeps track of how many retries have been attempted
    retry_count = 0
    
    try:
        # Check if the Flask server is running
        if not start_flask_server():
            print("Error: Could not start Flask server")
            return
        
        # Initialize the database
        print("Initializing database...")
        database.db.init_database()
        
        while retry_count < max_retries and not is_collection_complete():
            try:
                retry_count += 1
                print(f"\n=== Collection Attempt {retry_count}/{max_retries} ===")
                
                # Set up the WebDriver
                print("Setting up WebDriver...")
                if driver:
                    driver.quit()
                driver = setup_webdriver()
                
                # Start the collection process
                print("Starting fingerprint collection...")
                collected_this_round = collect_fingerprints(driver)
                
                if collected_this_round == 0:
                    print("No new traces collected this round")
                    
                # Check if we're done
                if is_collection_complete():
                    print("Collection target reached!")
                    break
                else:
                    current_counts = database.db.get_traces_collected()
                    remaining = sum(max(0, TRACES_PER_SITE - count) for count in current_counts.values())
                    print(f"Still need to collect {remaining} more traces")
                    
            except Exception as e:
                print(f"Error in collection attempt {retry_count}: {e}")
                traceback.print_exc()
                if retry_count < max_retries:
                    print(f"Retrying in 5 seconds...")
                    time.sleep(5)
        
        # Export the collected data to a JSON file
        print("Exporting data to JSON...")
        database.db.export_to_json(OUTPUT_PATH)
        
        # Final status report
        final_counts = database.db.get_traces_collected()
        total_collected = sum(final_counts.values())
        print(f"\n=== Collection Summary ===")
        print(f"Total traces collected: {total_collected}")
        for website, count in final_counts.items():
            print(f"  {website}: {count}/{TRACES_PER_SITE}")
        print(f"Data exported to: {OUTPUT_PATH}")
        
        if is_collection_complete():
            print("Collection completed successfully!")
        else:
            print("Collection incomplete - you may want to run the script again")
            
    except KeyboardInterrupt:
        print("\nCollection interrupted by user")
    except Exception as e:
        print(f"Fatal error in main: {e}")
        traceback.print_exc()
    finally:
        # Handle any exceptions and ensure the WebDriver is closed at the end
        if driver:
            print("Closing WebDriver...")
            driver.quit()
        
        # Save data one final time
        try:
            database.db.export_to_json(OUTPUT_PATH)
        except:
            pass

if __name__ == "__main__":
    main()