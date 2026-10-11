@echo off
cd /d %~dp0
python scripts\train.py --config configs\resnet18_focal.yaml --seed 2026
python scripts\train.py --config configs\densenet121_focal.yaml --seed 2026
python scripts\train.py --config configs\efficientnet_b0_focal.yaml --seed 2026
pause
