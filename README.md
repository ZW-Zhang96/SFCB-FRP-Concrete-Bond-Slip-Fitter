# Bond-Slip Modeler 📈

**Intelligent fitting tool for the bond stress-slip curve of SFCB/FRP bars and concrete.**

**Author:** Zhiwen Zhang

## 📖 Introduction
This repository provides a Python-based graphical user interface (GUI) application designed to automatically process, analyze, and fit experimental bond stress-slip data between concrete and Fiber Reinforced Polymer (FRP) or Steel-FRP Composite Bars (SFCB). 

Instead of manually guessing parameters, this tool intelligently identifies critical data points (such as peak stress, peak slip, and residual points) and applies advanced non-linear optimization to fit six widely used constitutive models simultaneously.

## ✨ Key Features
- **User-Friendly GUI:** Clean interface built with Tkinter, requiring no coding experience to operate.
- **Six Built-in Models:** Automatically fits and compares the following classical and advanced models:
  - Arnaud Model
  - Malvar Model
  - Hao Model
  - Gao Model
  - MBPE Model
  - Four-Stage Model
- **Intelligent Point Detection:** Automatically identifies micro-slip, peak slip/stress, and residual slip/stress from raw experimental scatter data.
- **Goodness-of-Fit Metrics:** Calculates Coefficient of Determination ($R^2$) and Root Mean Square Error (RMSE) for each model.
- **High-Quality Visualization:** Generates publication-ready comparative plots of experimental data versus fitted curves.
- **One-Click Export:** Exports model parameters, goodness-of-fit metrics, and point-by-point curve data directly to Excel (`.xlsx`) or CSV.

## 🛠️ Installation & Requirements
Ensure you have **Python 3.7 or higher** installed on your system. 

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/YourUsername/Bond-Slip-Modeler.git
   cd Bond-Slip-Modeler

🚀 How to Use
Launch the software:

code
Bash
python BondSlipModeler.py
Load Data: Click the Load Data button and select your experimental data file.

Run Fits: Click the Run All Fits button. The software will automatically analyze the curve and fit all six models.

Review Results:

Check the "Fitting Curves" tab to visually inspect the accuracy of the fits.

Check the "Detailed Parameters" tab to view the mathematical formulas and identified parameter values for each model.

Export: Click Export Results to save a comprehensive summary (metrics, parameters, and curve coordinates) to your computer.

📄 Input Data Format
The application expects a plain text file (.txt) containing the raw experimental data.

Format: Two columns separated by spaces or tabs.

Column 1: Slip (in mm).

Column 2: Bond Stress (in MPa).

Note: Do not include text headers in the file. Ensure values are non-negative.

Example (data.txt):

0.000   0.000
0.521   3.450
1.034   7.890
1.550   12.30
...

📝 License
This project is open-sourced under the MIT License. Feel free to use, modify, and distribute it for academic and engineering purposes.
