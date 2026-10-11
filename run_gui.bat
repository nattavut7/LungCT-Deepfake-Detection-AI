@echo off
cd /d %~dp0
python app\app_gradio.py --config configs\efficientnet_b0_focal.yaml --checkpoint outputs\checkpoints\efficientnet_b0_focal_seed2026\best.pt
pause
