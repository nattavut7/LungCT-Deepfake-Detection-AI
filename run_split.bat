@echo off
cd /d %~dp0
python scripts\split_patient_level.py --config configs\efficientnet_b0_focal.yaml
pause
