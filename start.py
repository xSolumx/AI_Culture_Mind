import subprocess

bat_file_path = 'C:\\Programming\\AI_Culture_Mind\\text-generation-webui\\start_windows.bat'

try:
    # Run the batch script using subprocess module in python.
    process = subprocess.Popen(['cmd.exe', '/c', bat_file_path], shell=True, stdout=subprocess.PIPE)

except Exception as e:
    print(f"Error: {e}")