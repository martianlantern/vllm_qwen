#!/usr/bin/env python3

import time
import threading
import random
import numpy as np
import pandas as pd
from io import BytesIO
from tqdm.auto import tqdm
from elevenlabs.client import ElevenLabs
import concurrent.futures
from collections import defaultdict
import queue

# Initialize Eleven Labs client
elevenlabs = ElevenLabs(
    api_key="sk_4902f02e03291772954c3f3a42ded11d174d09aca13bd0f6",
)

# Load audio data
df = pd.read_csv("samsung_test.csv")
audio_paths = ["audio_data/" + file.split("/")[-1] for file in df["file"]]
audio_paths = [audio_paths[idx] for idx in [0, 1, 2, 5]]

# Load audio data into memory for faster access
audio_data_cache = {}
for audio_path in audio_paths:
    with open(audio_path, "rb") as f:
        audio_data_cache[audio_path] = f.read()

def make_transcription_request(audio_data):
    """Make a single transcription request and measure latency."""
    start_time = time.time()
    
    try:
        audio_buffer = BytesIO(audio_data)
        transcription = elevenlabs.speech_to_text.convert(
            file=audio_buffer,
            model_id="scribe_v2",
            tag_audio_events=False,
            language_code="eng",
            diarize=False,
        )
        end_time = time.time()
        latency = (end_time - start_time) * 1000  # Convert to milliseconds
        return latency, True
    except Exception as e:
        end_time = time.time()
        latency = (end_time - start_time) * 1000
        print(f"Request failed: {e}")
        return latency, False

def generate_requests_at_rps(rps, duration_seconds, audio_data_cache, audio_paths):
    """Generate requests at specified RPS for given duration."""
    total_requests = int(rps * duration_seconds)
    interval = 1.0 / rps
    
    latencies = []
    successful_requests = 0
    failed_requests = 0
    
    # Use ThreadPoolExecutor for concurrent requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(50, total_requests)) as executor:
        futures = []
        
        for i in range(total_requests):
            # Randomly select an audio file
            audio_path = random.choice(audio_paths)
            audio_data = audio_data_cache[audio_path]
            
            # Submit request
            future = executor.submit(make_transcription_request, audio_data)
            futures.append(future)
            
            # Sleep to maintain RPS (except for last request)
            if i < total_requests - 1:
                time.sleep(interval)
        
        # Collect results
        for future in tqdm(concurrent.futures.as_completed(futures), 
                          desc=f"RPS {rps}", total=total_requests):
            try:
                latency, success = future.result(timeout=60)  # 60 second timeout
                latencies.append(latency)
                if success:
                    successful_requests += 1
                else:
                    failed_requests += 1
            except concurrent.futures.TimeoutError:
                print("Request timed out")
                failed_requests += 1
            except Exception as e:
                print(f"Request error: {e}")
                failed_requests += 1
    
    return latencies, successful_requests, failed_requests

def calculate_percentiles(latencies):
    """Calculate p50, p75, p95, p99 percentiles."""
    if not latencies:
        return 0, 0, 0, 0
    
    latencies = np.array(latencies)
    p50 = np.percentile(latencies, 50)
    p75 = np.percentile(latencies, 75)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    
    return p50, p75, p95, p99

def run_benchmark():
    """Run the complete latency benchmark."""
    # RPS settings to test
    rps_values = [4, 5, 6, 8, 10, 12]
    
    # Duration for each RPS test (in seconds)
    test_duration = 30  # 30 seconds per RPS setting
    
    results = []
    
    print("Starting latency benchmark...")
    print(f"Testing RPS values: {rps_values}")
    print(f"Test duration per RPS: {test_duration} seconds")
    print(f"Audio files: {len(audio_paths)}")
    
    for rps in rps_values:
        print(f"\n{'='*50}")
        print(f"Testing RPS: {rps}")
        print(f"{'='*50}")
        
        # Run the test
        latencies, successful, failed = generate_requests_at_rps(
            rps, test_duration, audio_data_cache, audio_paths
        )
        
        # Calculate percentiles
        p50, p75, p95, p99 = calculate_percentiles(latencies)
        
        # Store results
        results.append({
            'RPS': rps,
            'Total Requests': len(latencies) + failed,
            'Successful': successful,
            'Failed': failed,
            'Success Rate %': (successful / (successful + failed)) * 100 if (successful + failed) > 0 else 0,
            'p50': p50,
            'p75': p75,
            'p95': p95,
            'p99': p99,
            'Mean Latency': np.mean(latencies) if latencies else 0,
            'Std Latency': np.std(latencies) if latencies else 0
        })
        
        print(f"Results for RPS {rps}:")
        print(f"  Total requests: {len(latencies) + failed}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Success rate: {(successful / (successful + failed)) * 100:.1f}%" if (successful + failed) > 0 else "0.0%")
        print(f"  p50: {p50:.0f}ms")
        print(f"  p75: {p75:.0f}ms")
        print(f"  p95: {p95:.0f}ms")
        print(f"  p99: {p99:.0f}ms")
        
        # Small delay between RPS tests
        time.sleep(2)
    
    return results

def create_results_table(results):
    """Create and display results table similar to the image."""
    print(f"\n{'='*80}")
    print("LATENCY BENCHMARK RESULTS")
    print(f"{'='*80}")
    
    # Create DataFrame for better formatting
    df_results = pd.DataFrame(results)
    
    # Print summary table
    print("\nSUMMARY TABLE:")
    print("-" * 80)
    print(f"{'RPS':<6} {'p50':<12} {'p75':<12} {'p95':<12} {'p99':<12}")
    print("-" * 80)
    
    for _, row in df_results.iterrows():
        print(f"{int(row['RPS']):<6} "
              f"{int(row['p50']):<12} "
              f"{int(row['p75']):<12} "
              f"{int(row['p95']):<12} "
              f"{int(row['p99']):<12}")
    
    print("-" * 80)
    
    # Print detailed results
    print("\nDETAILED RESULTS:")
    print("-" * 100)
    for col in ['RPS', 'Total Requests', 'Successful', 'Failed', 'Success Rate %', 
                'p50', 'p75', 'p95', 'p99', 'Mean Latency', 'Std Latency']:
        if col in df_results.columns:
            df_results[col] = df_results[col].round(1) if df_results[col].dtype == 'float64' else df_results[col]
    
    print(df_results.to_string(index=False))
    
    # Save results to CSV
    output_file = "latency_benchmark_results.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    return df_results

if __name__ == "__main__":
    print("Eleven Labs Speech-to-Text Latency Benchmark")
    print("=" * 50)
    
    # Verify audio files exist
    missing_files = []
    for audio_path in audio_paths:
        try:
            with open(audio_path, "rb") as f:
                pass
        except FileNotFoundError:
            missing_files.append(audio_path)
    
    if missing_files:
        print(f"Error: Missing audio files: {missing_files}")
        exit(1)
    
    print(f"Loaded {len(audio_paths)} audio files into cache")
    
    # Run benchmark
    try:
        results = run_benchmark()
        df_results = create_results_table(results)
        
        print("\nBenchmark completed successfully!")
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"\nBenchmark failed with error: {e}")
        import traceback
        traceback.print_exc()
