@echo off
rem ============================================================
rem 医疗问答助手 - 双击启动入口（推荐）
rem 以 Bypass 执行策略调用 run_project.ps1
rem ============================================================
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_project.ps1"
