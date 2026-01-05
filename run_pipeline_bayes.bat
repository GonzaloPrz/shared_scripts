@echo off
:: Activate virtual environment
::call "C:\ProgramData\anaconda3\condabin\activate.bat" ml-env

::Execute Python scripts with arguments passed directly
"C:\Users\CNC Audio\.conda\envs\ml-env\python.exe" "C:\Users\CNC Audio\gonza\shared_scripts\train_models_bayes.py" %*
"C:\Users\CNC Audio\.conda\envs\ml-env\python.exe" "C:\Users\CNC Audio\gonza\shared_scripts\bootstrap_models_bayes.py"
"C:\Users\CNC Audio\.conda\envs\ml-env\python.exe" "C:\Users\CNC Audio\gonza\shared_scripts\train_final_model_bayes.py"
"C:\Users\CNC Audio\.conda\envs\ml-env\python.exe" "C:\Users\CNC Audio\gonza\shared_scripts\independent_tesT_set.py"
"C:\Users\CNC Audio\.conda\envs\ml-env\python.exe" "C:\Users\CNC Audio\gonza\shared_scripts\test_final_model_bayes.py"

:: Display used parameters for logging
echo Pipeline executed with arguments: %*