import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import re
from scipy.signal import butter, filtfilt

def filter_signal(signal_series, lowcut=0.17, highcut=0.4, fs=32, order=4):
    """
    Apply a Butterworth bandpass filter to remove high-frequency noise.
    Retains breathing frequency range (0.17-0.4 Hz).
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, signal_series.ffill().bfill())
    return pd.Series(filtered_data, index=signal_series.index)

def find_file_with_prefix(folder, prefix):
    """Find file with matching prefix in folder"""
    files_found = []
    
    for fname in os.listdir(folder):
        if fname.lower().endswith('.txt'):
            if prefix.lower() == "flow events":
                if fname.lower().startswith("flow events"):
                    files_found.append(fname)
            elif prefix.lower() == "sleep profile":
                # Look for files starting with "Sleep" for sleep profile
                if fname.lower().startswith("sleep"):
                    files_found.append(fname)
            else:
                fname_lower = fname.lower()
                if fname_lower.startswith("flow events") or fname_lower.startswith("sleep"):
                    continue
                first_word = fname.split()[0].lower() if fname.split() else ""
                if first_word == prefix.lower():
                    files_found.append(fname)
    
    if not files_found:
        raise FileNotFoundError(f"No file matching '{prefix}' in {folder}")
    
    return os.path.join(folder, files_found[0])

def load_signal_file(file_path, signal_name):
    """Load and parse signal file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    start_index = next(i for i, line in enumerate(lines) if line.strip().startswith("Data:")) + 1
    data_lines = lines[start_index:]

    timestamps, values = [], []
    for line in data_lines:
        if ";" not in line:
            continue
        try:
            time_str, value_str = line.strip().split(";")
            timestamp = datetime.strptime(time_str.strip(), "%d.%m.%Y %H:%M:%S,%f")
            value_str = value_str.strip().replace(",", ".")
            numeric_match = re.match(r'^-?\d+\.?\d*', value_str)
            if numeric_match:
                value = float(numeric_match.group())
            else:
                continue
            timestamps.append(timestamp)
            values.append(value)
        except:
            continue

    return pd.DataFrame({signal_name: values}, index=pd.to_datetime(timestamps))

def load_sleep_profile_file(file_path):
    """Load and parse sleep profile file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Data starts on 8th line (index 7)
    start_index = 7
    
    if len(lines) <= start_index:
        raise ValueError(f"File has only {len(lines)} lines, expected data starting at line 8")
    
    data_lines = lines[start_index:]

    sleep_stages = []
    for line_num, line in enumerate(data_lines):
        if ";" not in line or line.strip() == "":
            continue
        try:
            parts = line.strip().split(";")
            if len(parts) < 2:
                continue
                
            time_str = parts[0].strip()
            stage_str = parts[1].strip()
            
            # Parse timestamp in format: 30.05.2024 20:59:00,000
            timestamp = datetime.strptime(time_str, "%d.%m.%Y %H:%M:%S,%f")
            
            sleep_stages.append((timestamp, stage_str))
            
        except Exception as e:
            print(f"Error parsing line {start_index + line_num + 1}: {line.strip()} - {e}")
            continue

    return pd.DataFrame(sleep_stages, columns=["start_time", "sleep_stage"])

def calculate_overlap_percentage(window_start, window_end, event_start, event_end):
    """Calculate percentage overlap between window and event"""
    overlap_start = max(window_start, event_start)
    overlap_end = min(window_end, event_end)
    
    if overlap_start >= overlap_end:
        return 0.0
    
    overlap_duration = (overlap_end - overlap_start).total_seconds()
    window_duration = (window_end - window_start).total_seconds()
    
    return overlap_duration / window_duration

def assign_window_label(window_start, window_end, df_sleep_profile):
    """
    Assign sleep stage label to window based on overlap with sleep profile annotations.
    Modified from breathing irregularity framework to work with sleep stages.
    """
    if len(df_sleep_profile) == 0:
        return "Unknown"
    
    # Sort sleep profile by timestamp
    df_sleep_profile = df_sleep_profile.sort_values('start_time').reset_index(drop=True)
    
    # Find sleep stages that overlap with the window
    overlapping_stages = []
    
    for i in range(len(df_sleep_profile)):
        stage_start = df_sleep_profile.iloc[i]['start_time']
        sleep_stage = df_sleep_profile.iloc[i]['sleep_stage']
        
        # Determine stage end time (sleep stages are duration-based)
        if i + 1 < len(df_sleep_profile):
            stage_end = df_sleep_profile.iloc[i + 1]['start_time']
        else:
            # Last stage - extend to window end or reasonable duration
            stage_end = max(window_end, stage_start + pd.Timedelta(minutes=30))
        
        # Calculate overlap percentage
        overlap_pct = calculate_overlap_percentage(window_start, window_end, stage_start, stage_end)
        
        if overlap_pct > 0.5:  # More than 50% overlap (same as breathing irregularity)
            overlapping_stages.append((sleep_stage, overlap_pct))
    
    if not overlapping_stages:
        # Find closest sleep stage if no overlap (fallback)
        time_diffs = abs(df_sleep_profile['start_time'] - window_start)
        closest_idx = time_diffs.idxmin()
        return df_sleep_profile.iloc[closest_idx]['sleep_stage']
    
    # Return stage with maximum overlap
    best_stage = max(overlapping_stages, key=lambda x: x[1])
    return best_stage[0]

def create_dataset(in_dir, out_dir):
    """
    Create dataset from all participants in input directory.
    30-second windows with 50% overlap.
    Modified from breathing irregularity framework to work with sleep stages.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Find all participant folders
    participant_folders = []
    for item in os.listdir(in_dir):
        item_path = os.path.join(in_dir, item)
        if os.path.isdir(item_path):
            participant_folders.append(item_path)
    
    if not participant_folders:
        print(f"No participant folders found in {in_dir}")
        return
    
    print(f"Found {len(participant_folders)} participant folders")
    
    all_windows = []
    
    for participant_path in participant_folders:
        participant_id = os.path.basename(participant_path)
        print(f"\nProcessing participant: {participant_id}")
        
        try:
            # Load signals (same as breathing irregularity framework)
            flow_file = find_file_with_prefix(participant_path, "Flow")
            df_flow = load_signal_file(flow_file, "flow")
            
            thorac_file = find_file_with_prefix(participant_path, "Thorac")
            df_thor = load_signal_file(thorac_file, "thoracic")
            
            spo2_file = find_file_with_prefix(participant_path, "SPO2")
            df_spo2 = load_signal_file(spo2_file, "spo2")
            
            # Load sleep profile instead of flow events (KEY CHANGE)
            sleep_profile_file = find_file_with_prefix(participant_path, "Sleep Profile")
            df_sleep_profile = load_sleep_profile_file(sleep_profile_file)
            
            # Apply filtering to Flow and Thoracic signals (same as breathing irregularity)
            print("Applying bandpass filter (0.17-0.4 Hz) to Flow and Thoracic signals")
            df_flow['flow'] = filter_signal(df_flow['flow'], fs=32)
            df_thor['thoracic'] = filter_signal(df_thor['thoracic'], fs=32)
            # SpO2 is not filtered as per assignment
            
            # Combine signals (same as breathing irregularity framework)
            df_all = df_flow.join(df_thor, how='outer').join(df_spo2, how='outer')
            df_all = df_all.sort_index().interpolate().ffill().bfill()
            
            # Create 30-second windows with 50% overlap (same as breathing irregularity)
            window_size = pd.Timedelta(seconds=30)
            step_size = pd.Timedelta(seconds=15)  # 50% overlap
            
            start_time = df_all.index.min()
            end_time = df_all.index.max()
            
            current_time = start_time
            window_id = 0
            
            while current_time + window_size <= end_time:
                window_end = current_time + window_size
                
                # Extract window data (same as breathing irregularity)
                window_data = df_all.loc[current_time:window_end]
                
                if len(window_data) > 0:
                    # Assign label based on sleep stage overlap (MODIFIED)
                    label = assign_window_label(current_time, window_end, df_sleep_profile)
                    
                    # Create window record (same structure as breathing irregularity)
                    window_record = {
                        'participant_id': participant_id,
                        'window_id': window_id,
                        'start_time': current_time,
                        'end_time': window_end,
                        'label': label,
                        'flow_signal': window_data['flow'].tolist(),
                        'thoracic_signal': window_data['thoracic'].tolist(),
                        'spo2_signal': window_data['spo2'].tolist()
                    }
                    
                    all_windows.append(window_record)
                    window_id += 1
                
                current_time += step_size
            
            print(f"Created {window_id} windows for {participant_id}")
            
        except Exception as e:
            print(f"Error processing {participant_id}: {e}")
    
    # Convert to DataFrame and save (same as breathing irregularity framework)
    if all_windows:
        df_dataset = pd.DataFrame(all_windows)
        
        # Save as CSV (same format as breathing irregularity)
        output_file = os.path.join(out_dir, "sleep_stage_dataset.csv")
        
        # For CSV, we need to handle the signal lists differently
        # Convert signal lists to string representation for CSV compatibility
        df_csv = df_dataset.copy()
        for col in ['flow_signal', 'thoracic_signal', 'spo2_signal']:
            df_csv[col] = df_csv[col].apply(lambda x: ','.join(map(str, x)))
        
        df_csv.to_csv(output_file, index=False)
        
        print(f"\nDataset saved to: {output_file}")
        print(f"Total windows: {len(df_dataset)}")
        
        # Show label distribution (modified for sleep stages)
        label_counts = df_dataset['label'].value_counts()
        print(f"\nSleep stage distribution:")
        for label, count in label_counts.items():
            percentage = (count / len(df_dataset)) * 100
            print(f"  {label}: {count} windows ({percentage:.1f}%)")
            
        print(f"\nDataset format: CSV")
        print(f"Reason: CSV format chosen for easy inspection, sharing, and compatibility")
        print(f"with various analysis tools. Signal data stored as comma-separated values.")
    
    else:
        print("No windows were created")

def main():
    parser = argparse.ArgumentParser(description="Create sleep stage dataset from sleep study data")
    parser.add_argument("-in_dir", type=str, required=True, help="Input directory containing participant folders")
    parser.add_argument("-out_dir", type=str, required=True, help="Output directory for dataset")
    
    args = parser.parse_args()
    
    create_dataset(args.in_dir, args.out_dir)

if __name__ == "__main__":
    main()