@echo off
:: Activate virtual environment
::call "C:\ProgramData\anaconda3\condabin\activate.bat" gonza-env

::Execute Python scripts with arguments passed directly
"C:\Users\CNC Audio\.conda\envs\gonza-env\python.exe" "C:\Users\CNC Audio\gonza\scripts\shared\train_models_bayes.py" %*
"C:\Users\CNC Audio\.conda\envs\gonza-env\python.exe" "C:\Users\CNC Audio\gonza\scripts\shared\bootstrap_models_bayes.py"
"C:\Users\CNC Audio\.conda\envs\gonza-env\python.exe" "C:\Users\CNC Audio\gonza\scripts\shared\train_final_model_bayes.py"
"C:\Users\CNC Audio\.conda\envs\gonza-env\python.exe" "C:\Users\CNC Audio\gonza\scripts\shared\test_final_model_bayes.py"

:: Display used parameters for logging
echo Pipeline executed with arguments: %*