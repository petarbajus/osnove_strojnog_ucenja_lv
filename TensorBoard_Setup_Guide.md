# TensorBoard Setup Guide

This guide explains how to enable and run TensorBoard on Windows (using PowerShell) to visualize TensorFlow/Keras training logs. It assumes Python 3.13 is installed globally and you're working in a project directory like `c:\osu_test_2\osnove_strojnog_ucenja_lv`.

## Prerequisites
- Python 3.13 installed (or another version; note that Python 3.13 has compatibility issues with some TensorBoard versions due to the removal of `pkg_resources`).
- Your project logs are in a directory like `lv9/logs` (with subfolders for different runs, e.g., `cnn`, `cnn_dropout`).

## Step 1: Create a Virtual Environment
To isolate dependencies and avoid conflicts with your global Python installation:
1. Open PowerShell and navigate to your project root:
   ```
   cd c:\osu_test_2\osnove_strojnog_ucenja_lv
   ```
2. Create a virtual environment named `venv`:
   ```
   python -m venv venv
   ```
   This creates a `venv` folder with its own Python interpreter and package space.

## Step 2: Activate the Virtual Environment
1. Activate the virtual environment:
   ```
   .\venv\Scripts\activate.ps1
   ```
   Your prompt should now show `(venv)` at the beginning, indicating it's active. All subsequent commands will use this environment.

## Step 3: Install TensorFlow (Includes TensorBoard)
TensorBoard is bundled with TensorFlow, so installing TensorFlow ensures you have a compatible version.
1. Install TensorFlow:
   ```
   pip install tensorflow
   ```
   This may take a few minutes as it's a large package. If you prefer only TensorBoard (smaller install), you can try `pip install tensorboard`, but it may fail on Python 3.13 (see troubleshooting below).

## Step 4: Fix Compatibility Issues (If Using Python 3.13)
Python 3.13 removed the `pkg_resources` module from `setuptools`, causing import errors in TensorBoard. If you encounter `ModuleNotFoundError: No module named 'pkg_resources'`:
1. Downgrade `setuptools` to a version that includes `pkg_resources`:
   ```
   pip install setuptools==69.5.1
   ```
2. Verify the fix:
   ```
   python -c "import pkg_resources"
   ```
   It should succeed (with a deprecation warning, which is fine).

## Step 5: Run TensorBoard
1. Start TensorBoard, pointing to your logs directory:
   ```
   .\venv\Scripts\tensorboard.exe --logdir=lv9/logs
   ```
   - Replace `lv9/logs` with your actual logs path (e.g., `lv9/logs/cnn` for a specific run).
   - TensorBoard will start a local server (usually on `http://localhost:6006`).
2. Open `http://localhost:6006` in your browser to view the logs:
   - Use tabs like "SCALARS" for metrics (e.g., loss, accuracy), "GRAPHS" for the model architecture, and "HISTOGRAMS" for weight distributions.
   - If you have multiple runs, select them from the "Runs" dropdown on the left.

## Step 6: Stop TensorBoard
- In the PowerShell terminal where TensorBoard is running, press `Ctrl+C` to stop the server.
- Deactivate the virtual environment (optional):
  ```
  deactivate
  ```

## Troubleshooting Tips
- **Port in use**: If `localhost:6006` is busy, specify a different port: `tensorboard --logdir=lv9/logs --port=6007`.
- **No logs showing**: Ensure your logs are in TensorFlow's event file format (`.tfevents` files). If not, check your training script for proper logging (e.g., `tf.keras.callbacks.TensorBoard`).
- **Python version issues**: If Python 3.13 causes problems, consider using Python 3.11 or 3.12 for better compatibility. You can install multiple versions via the official installer and specify the version for the venv (e.g., `py -3.11 -m venv venv`).
- **Permission errors**: Run PowerShell as Administrator if you get access issues.
- **Virtual environment not activating**: Ensure you're using `activate.ps1` for PowerShell (or `activate.bat` for Command Prompt).

This setup ensures TensorBoard runs reliably in an isolated environment.