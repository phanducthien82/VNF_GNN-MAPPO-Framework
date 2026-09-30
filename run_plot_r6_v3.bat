@echo off
setlocal

REM Run from the experiment directory:
REM D:\Papers\ThienPD\Conf\CITA2027\Codes\experiment

python plot_r6_complete_experiment_v3.py

if errorlevel 1 (
    echo.
    echo [ERROR] Plot generation failed.
    pause
    exit /b 1
)

echo.
echo [OK] R6 figures and tables generated.
echo Output: figures_r6_complete_v3
pause
