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
            else:
                fname_lower = fname.lower()
                if fname_lower.startswith("flow events"):
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

def load_event_file(event_path):
    """Load and parse event annotations file"""
    with open(event_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    data_lines = [line for line in lines if "-" in line and ";" in line]
    events = []
    for line in data_lines:
        try:
            time_part, _, label, _ = line.strip().split(";")
            start_str, end_str = time_part.strip().split("-")
            start = datetime.strptime(start_str.strip(), "%d.%m.%Y %H:%M:%S,%f")
            end = datetime.strptime(end_str.strip(), "%H:%M:%S,%f")
            end = end.replace(year=start.year, month=start.month, day=start.day)
            events.append((start, end, label))
        except:
            continue

    return pd.DataFrame(events, columns=["start_time", "end_time", "event_type"])

def calculate_overlap_percentage(window_start, window_end, event_start, event_end):
    """Calculate percentage overlap between window and event"""
    overlap_start = max(window_start, event_start)
    overlap_end = min(window_end, event_end)
    
    if overlap_start >= overlap_end:
        return 0.0
    
    overlap_duration = (overlap_end - overlap_start).total_seconds()
    window_duration = (window_end - window_start).total_seconds()
    
    return overlap_duration / window_duration

def assign_window_label(window_start, window_end, df_events):
    """
    Assign label to window based on overlap with events.
    Priority: Obstructive Apnea > Hypopnea > Normal
    """
    target_events = ["Hypopnea", "Obstructive Apnea"]
    
    # Check for overlaps with target events
    overlaps = []
    for _, event in df_events.iterrows():
        if event['event_type'] in target_events:
            overlap_pct = calculate_overlap_percentage(
                window_start, window_end, 
                event['start_time'], event['end_time']
            )
            if overlap_pct > 0.5:  # More than 50% overlap
                overlaps.append((event['event_type'], overlap_pct))
    
    if not overlaps:
        return "Normal"
    
    # Priority: Obstructive Apnea > Hypopnea
    for event_type, _ in overlaps:
        if event_type == "Obstructive Apnea":
            return "Obstructive Apnea"
    
    for event_type, _ in overlaps:
        if event_type == "Hypopnea":
            return "Hypopnea"
    
    return "Normal"

def create_dataset(in_dir, out_dir):
    """
    Create dataset from all participants in input directory.
    30-second windows with 50% overlap.
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
            # Load signals
            flow_file = find_file_with_prefix(participant_path, "Flow")
            df_flow = load_signal_file(flow_file, "flow")
            
            thorac_file = find_file_with_prefix(participant_path, "Thorac")
            df_thor = load_signal_file(thorac_file, "thoracic")
            
            spo2_file = find_file_with_prefix(participant_path, "SPO2")
            df_spo2 = load_signal_file(spo2_file, "spo2")
            
            # Load events
            events_file = find_file_with_prefix(participant_path, "Flow Events")
            df_events = load_event_file(events_file)
            
            # Apply filtering to Flow and Thoracic signals
            print("Applying bandpass filter (0.17-0.4 Hz) to Flow and Thoracic signals")
            df_flow['flow'] = filter_signal(df_flow['flow'], fs=32)
            df_thor['thoracic'] = filter_signal(df_thor['thoracic'], fs=32)
            # SpO2 is not filtered as per assignment
            
            # Combine signals
            df_all = df_flow.join(df_thor, how='outer').join(df_spo2, how='outer')
            df_all = df_all.sort_index().interpolate().ffill().bfill()
            
            # Create 30-second windows with 50% overlap
            window_size = pd.Timedelta(seconds=30)
            step_size = pd.Timedelta(seconds=15)  # 50% overlap
            
            start_time = df_all.index.min()
            end_time = df_all.index.max()
            
            current_time = start_time
            window_id = 0
            
            while current_time + window_size <= end_time:
                window_end = current_time + window_size
                
                # Extract window data
                window_data = df_all.loc[current_time:window_end]
                
                if len(window_data) > 0:
                    # Assign label based on event overlap
                    label = assign_window_label(current_time, window_end, df_events)
                    
                    # Create window record
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
    
    # Convert to DataFrame and save
    if all_windows:
        df_dataset = pd.DataFrame(all_windows)
        
        # Save as CSV
        output_file = os.path.join(out_dir, "breathing_dataset.csv")
        
        # Converting signal lists to string representation for CSV compatibility
        df_csv = df_dataset.copy()
        for col in ['flow_signal', 'thoracic_signal', 'spo2_signal']:
            df_csv[col] = df_csv[col].apply(lambda x: ','.join(map(str, x)))
        
        df_csv.to_csv(output_file, index=False)
        
        print(f"\nDataset saved to: {output_file}")
        print(f"Total windows: {len(df_dataset)}")
        
        # Show label distribution
        label_counts = df_dataset['label'].value_counts()
        print(f"\nLabel distribution:")
        for label, count in label_counts.items():
            print(f"  {label}: {count} windows")
            
        print(f"\nDataset format: CSV")
        print(f"Reason: CSV format chosen for easy inspection, sharing, and compatibility")
        print(f"with various analysis tools. Signal data stored as comma-separated values.")
    
    else:
        print("No windows were created")

def main():
    parser = argparse.ArgumentParser(description="Create dataset from sleep study data")
    parser.add_argument("-in_dir", type=str, required=True, help="Input directory containing participant folders")
    parser.add_argument("-out_dir", type=str, required=True, help="Output directory for dataset")
    
    args = parser.parse_args()
    
    create_dataset(args.in_dir, args.out_dir)

if __name__ == "__main__":
    main()