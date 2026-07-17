@echo off
chcp 65001 >nul
echo ========================================
echo 虚拟环境重建脚本
echo ========================================
echo.

:: 1. 退出当前虚拟环境（如果存在）
echo [1/6] 退出当前虚拟环境...
if defined VIRTUAL_ENV (
    echo 正在退出虚拟环境...
    deactivate
    echo 已退出
) else (
    echo 未检测到激活的虚拟环境
)
echo.
timeout /t 2 /nobreak >nul

:: 2. 删除旧的虚拟环境文件夹
echo [2/6] 删除旧的虚拟环境文件夹...
if exist ..\.venv (
    echo 正在删除 ..\.venv 文件夹...
    rmdir /s /q ..\.venv
    echo 删除完成
) else (
    echo ..\.venv 文件夹不存在，跳过删除
)
echo.
timeout /t 2 /nobreak >nul

:: 3. 创建新的虚拟环境
echo [3/6] 创建新的虚拟环境...
python -m venv ..\.venv
if %errorlevel% neq 0 (
    echo 错误：创建虚拟环境失败，请检查 Python 是否已安装
    pause
    exit /b 1
)
echo 虚拟环境创建成功
echo.
timeout /t 2 /nobreak >nul

:: 4. 激活虚拟环境（使用 CMD 兼容的 bat 脚本）
echo [4/6] 激活虚拟环境...
call ..\.venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo 错误：激活虚拟环境失败
    pause
    exit /b 1
)
echo 虚拟环境已激活
echo.
timeout /t 2 /nobreak >nul

:: 5. 升级 pip
echo [5/6] 升级 pip 到最新版本...
python -m pip install --upgrade pip
echo.
timeout /t 2 /nobreak >nul

:: 6. 安装依赖
echo [6/6] 安装 requirements.txt 中的依赖...
echo 正在安装，请稍候...
echo.
pip install -r ..\requirements.txt

if %errorlevel% neq 0 (
    echo.
    echo 错误：依赖安装失败，请检查 requirements.txt 内容
    pause
    exit /b 1
)

echo.
echo ========================================
echo 环境重建完成！
echo ========================================
echo.
echo 当前虚拟环境已激活，可以运行 Streamlit 前端：
echo       streamlit run ..\app.py
echo.
echo 提示：如果再次打开终端，需要手动激活虚拟环境：
echo       方式1 (CMD): .venv\Scripts\activate.bat
echo       方式2 (PowerShell): .venv\Scripts\Activate.ps1
echo.
pause