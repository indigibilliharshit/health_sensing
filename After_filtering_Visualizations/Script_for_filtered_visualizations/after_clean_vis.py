import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.dates as mdates
from datetime import datetime
import re
from scipy.signal import butter, filtfilt

def filter_signal(signal_series, lowcut=0.17, highcut=0.4, fs=32, order=4):
    """
    Apply a Butterworth bandpass filter to a signal.
    Parameters:
    - signal_series: Pandas Series (indexed by timestamp)
    - lowcut: Low cutoff frequency in Hz (default 0.17 Hz)
    - highcut: High cutoff frequency in Hz (default 0.4 Hz)
    - fs: Sampling rate of the signal (default 32 Hz)
    - order: Filter order (default 4)
    Returns:
    - Filtered Pandas Series
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, signal_series.ffill().bfill())
    return pd.Series(filtered_data, index=signal_series.index)

def find_file_with_prefix(folder, prefix):
    """Enhanced file finder that handles different naming conventions"""
    files_found = []
    
    for fname in os.listdir(folder):
        if fname.lower().endswith('.txt'):
            # Special case for "Flow Events" - match exactly
            if prefix.lower() == "flow events":
                if fname.lower().startswith("flow events"):
                    files_found.append(fname)
            else:
                # For all other files, match by first word only
                # But exclude "Flow Events" when searching for "Flow"
                fname_lower = fname.lower()
                
                # Skip "Flow Events" files when looking for other types
                if fname_lower.startswith("flow events"):
                    continue
                
                # Extract first word from filename
                first_word = fname.split()[0].lower() if fname.split() else ""
                
                if first_word == prefix.lower():
                    files_found.append(fname)
    
    if not files_found:
        raise FileNotFoundError(f"No file matching '{prefix}' in {folder}")
    
    if len(files_found) > 1:
        print(f"Multiple files found for '{prefix}': {files_found}")
        print(f"Using: {files_found[0]}")
    
    return os.path.join(folder, files_found[0])

def load_signal_file(file_path, signal_name):
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

def calculate_smart_ylim(data, default_min=None, default_max=None, padding_factor=0.1):
    if data.empty or data.isnull().all():
        return (default_min or 0, default_max or 100)
    
    data_min = data.min()
    data_max = data.max()
    data_range = data_max - data_min
    
    # Handle case where all values are identical or very close
    if data_range == 0 or data_range < 1e-10:
        # Create a small range around the value
        center = data_min
        if abs(center) > 1:
            # For larger values, use 10% of the value as range
            range_size = abs(center) * 0.1
        else:
            # For small values, use a fixed small range
            range_size = 1.0
        
        y_min = center - range_size
        y_max = center + range_size
    else:
        padding = data_range * padding_factor
        y_min = data_min - padding
        y_max = data_max + padding
    
    if default_min is not None:
        y_min = min(y_min, default_min)
    if default_max is not None:
        y_max = max(y_max, default_max)
    
    return (y_min, y_max)

def generate_smart_ticks(y_min, y_max, target_ticks=4):
    y_range = y_max - y_min
    
    # Handle edge cases
    if y_range == 0 or y_range < 1e-10:
        # If range is zero or very small, create simple ticks around the center
        center = (y_min + y_max) / 2
        return [center - 1, center, center + 1]
    
    rough_interval = y_range / target_ticks
    
    # Handle very small intervals
    if rough_interval < 1e-10:
        return [y_min, (y_min + y_max) / 2, y_max]
    
    magnitude = 10 ** np.floor(np.log10(rough_interval))
    normalized = rough_interval / magnitude
    
    if normalized <= 1:
        nice_interval = magnitude
    elif normalized <= 2:
        nice_interval = 2 * magnitude
    elif normalized <= 5:
        nice_interval = 5 * magnitude
    else:
        nice_interval = 10 * magnitude
    
    first_tick = np.ceil(y_min / nice_interval) * nice_interval
    ticks = []
    tick = first_tick
    while tick <= y_max:
        ticks.append(tick)
        tick += nice_interval
    
    # Ensure we have at least 2 ticks
    if len(ticks) < 2:
        ticks = [y_min, y_max]
    
    return ticks

def plot_window(start, end, df_all, event_df, participant_id):
    fig, axs = plt.subplots(3, 1, figsize=(24, 8), sharex=True)
    fig.subplots_adjust(hspace=0.2, left=0.04, right=0.99, top=0.90, bottom=0.20)

    df_win = df_all.loc[start:end]

    # Plot 1: Nasal Flow
    axs[0].plot(df_win.index, df_win['flow'], color='blue', linewidth=0.8, label='Nasal Flow')
    axs[0].set_ylabel("Nasal Flow (L/min)", fontsize=11)
    axs[0].grid(True, alpha=0.5)
    axs[0].legend(loc='upper right', fontsize=10)
    
    flow_min, flow_max = calculate_smart_ylim(df_win['flow'], default_min=-130, default_max=130)
    axs[0].set_ylim(flow_min, flow_max)
    flow_ticks = generate_smart_ticks(flow_min, flow_max, target_ticks=4)
    axs[0].set_yticks(flow_ticks)
    axs[0].tick_params(axis='y', labelsize=10)

    # Plot 2: Thoracic Movement
    axs[1].plot(df_win.index, df_win['thoracic'], color='orange', linewidth=0.8, label='Thoracic/Abdominal Resp.')
    axs[1].set_ylabel("Resp. Amplitude", fontsize=11)
    axs[1].grid(True, alpha=0.5)
    axs[1].legend(loc='upper right', fontsize=10)
    
    thoracic_min, thoracic_max = calculate_smart_ylim(df_win['thoracic'], padding_factor=0.15)
    axs[1].set_ylim(thoracic_min, thoracic_max)
    thoracic_ticks = generate_smart_ticks(thoracic_min, thoracic_max, target_ticks=4)
    axs[1].set_yticks(thoracic_ticks)
    axs[1].tick_params(axis='y', labelsize=10)

    # Plot 3: SpO2
    axs[2].plot(df_win.index, df_win['spo2'], color='green', linewidth=0.8, label='SpO2')
    axs[2].set_ylabel("SpO₂ (%)", fontsize=11)
    axs[2].set_xlabel("Time", fontsize=11)
    axs[2].grid(True, alpha=0.5)
    axs[2].legend(loc='upper right', fontsize=10)
    
    spo2_data = df_win['spo2'].dropna()
    if not spo2_data.empty:
        spo2_min = spo2_data.min()
        spo2_max = spo2_data.max()
        plot_min = max(85, spo2_min - 1)
        plot_max = min(100, spo2_max + 1)
        axs[2].set_ylim(plot_min, plot_max)
        
        spo2_range = plot_max - plot_min
        if spo2_range <= 3:
            tick_interval = 0.5
        elif spo2_range <= 6:
            tick_interval = 1.0
        else:
            tick_interval = 2.0
        
        first_tick = np.ceil(plot_min / tick_interval) * tick_interval
        spo2_ticks = []
        tick = first_tick
        while tick <= plot_max:
            spo2_ticks.append(tick)
            tick += tick_interval
        
        axs[2].set_yticks(spo2_ticks)
    else:
        axs[2].set_ylim(93, 100)
        axs[2].set_yticks([93, 94, 95, 96, 97, 98, 99, 100])
    
    axs[2].tick_params(axis='y', labelsize=10)

    # Add event annotations (removed arousal and central apnea)
    for _, row in event_df.iterrows():
        if row['end_time'] >= start and row['start_time'] <= end:
            event_start = max(row['start_time'], start)
            event_end = min(row['end_time'], end)

            event_type = row['event_type'].strip().lower()
            
            # Define colors for different event types (removed arousal and central apnea)
            if event_type == "hypopnea":
                event_color = 'yellow'
                text_color = 'black'
            elif event_type == "obstructive apnea":
                event_color = 'red'
                text_color = 'white'
            elif event_type == "mixed apnea":
                event_color = 'purple'
                text_color = 'white'
            elif "body" in event_type:
                event_color = 'orange'
                text_color = 'black'
            else:
                event_color = 'gray'
                text_color = 'white'

            axs[0].axvspan(event_start, event_end, color=event_color, alpha=0.3)
            axs[0].text(
                event_start + (event_end - event_start) / 2,
                axs[0].get_ylim()[1] * 0.9,
                row['event_type'],
                color=text_color,
                fontsize=8,
                rotation=0,
                ha='center',
                va='top',
                bbox=dict(facecolor=event_color, alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3')
            )

    # Format x-axis
    for i, ax in enumerate(axs):
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        if i < 2:
            ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
        else:
            ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %H:%M:%S'))
            ax.xaxis.set_major_locator(mdates.SecondLocator(interval=5))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=90, ha='center', va='top', fontsize=10)

    title_start = start.strftime('%Y-%m-%d %H:%M')
    title_end = end.strftime('%Y-%m-%d %H:%M')
    fig.suptitle(f"{participant_id} - {title_start} to {title_end}", fontsize=16, y=0.95)
    
    return fig

def main(participant_path):
    participant_id = os.path.basename(participant_path.rstrip('/\\'))
    print(f"Processing: {participant_id}")

    # Load files with the enhanced file finder
    
    try:
        flow_file = find_file_with_prefix(participant_path, "Flow")
        print(f"Flow file: {os.path.basename(flow_file)}")
        df_flow = load_signal_file(flow_file, "flow")
        # Apply filter to flow signal
        df_flow['flow'] = filter_signal(df_flow['flow'])
    except Exception as e:
        print(f"Error with flow file: {e}")
        return

    try:
        thorac_file = find_file_with_prefix(participant_path, "Thorac")
        print(f"Thoracic file: {os.path.basename(thorac_file)}")
        df_thor = load_signal_file(thorac_file, "thoracic")
        # Apply filter to thoracic signal
        df_thor['thoracic'] = filter_signal(df_thor['thoracic'])
    except Exception as e:
        print(f"Error with thoracic file: {e}")
        return

    try:
        spo2_file = find_file_with_prefix(participant_path, "SPO2")
        print(f"SpO2 file: {os.path.basename(spo2_file)}")
        df_spo2 = load_signal_file(spo2_file, "spo2")
    except Exception as e:
        print(f"Error with SpO2 file: {e}")
        return

    try:
        events_file = find_file_with_prefix(participant_path, "Flow Events")
        print(f"Events file: {os.path.basename(events_file)}")
        df_events = load_event_file(events_file)
    except Exception as e:
        print(f"Error with events file: {e}")
        return

    df_all = df_flow.join(df_thor, how='outer').join(df_spo2, how='outer')
    df_all = df_all.sort_index().interpolate().ffill().bfill()

    os.makedirs("After Cleaning Visualizations", exist_ok=True)
    pdf_path = f"After Cleaning Visualizations/{participant_id}_After_Cleaning_visualization.pdf"

    start_time = df_all.index.min().floor('min')
    end_time = df_all.index.max().ceil('min')
    window = pd.Timedelta(minutes=5)

    print(f"Creating visualization: {pdf_path}")
    with PdfPages(pdf_path) as pdf:
        while start_time + window <= end_time:
            fig = plot_window(start_time, start_time + window, df_all, df_events, participant_id)
            pdf.savefig(fig, dpi=200, bbox_inches='tight', pad_inches=0.1, orientation='landscape')
            plt.close(fig)
            start_time += window

    print(f"Saved PDF to {pdf_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-name", type=str, required=True, help="Path to participant folder (e.g., Data/AP05)")
    args = parser.parse_args()
    main(args.name)