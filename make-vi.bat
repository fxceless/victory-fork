::Rust tools: rustup, rustc, cargo
::setx RUSTUP_HOME "D:\Progs\IDE\Rust\.rustup"
::setx CARGO_HOME "D:\Progs\IDE\Rust\.cargo"
::Get MSVC Build Tools (vs_BuildTools.exe)(VSBuildTools): https://aka.ms/vs/17/release.ltsc.17.2/vs_buildtools.exe
:: Need Installer Packages:
:: -- MSVC v143 VS 2022 C++ x64/86 Build Tools
:: -- Windows 10 SDK

@echo off
chcp 65001 >nul
cd /d "%~dp0"

::set "VS_BuildTools="

echo ========================================================
echo   VIctory - One Click Project Build Script ^| (powered by ncpp Project)
echo ========================================================
echo.

set mode=
if "%~1"=="DEBUG" (set "mode=%~1")
if "%~1"=="--help" (set "mode=help")
if "%1"=="RELEASE" (shift)
if "%1"=="DEF" (shift)
if not "%mode%"=="" (shift)
if "%mode%"=="help" echo Syntax: make-vi.bat [mode(DEBUG/RELEASE/DEF)] [Run tests(TESTS/SKIP_TESTS)] & exit /b 0

set SKIP_TESTS=
if "%~1"=="SKIP_TESTS" (set "SKIP_TESTS=1")
if "%1"=="TESTS" (shift)
if not "%SKIP_TESTS%"=="" (shift)

where cargo >nul 2>nul
if %errorlevel% neq 0 (echo [ERROR] Cargo not found. Install Rust ^(rustup^) with x86_64-pc-windows-msvc: https://rust-lang.org/tools/install/ & pause & exit /b 1)
::where link >nul 2>nul
::if %errorlevel% neq 0 (echo [INFO] Environment vcvars64.bat initialize... & call "%VS_BuildTools%\VC\Auxiliary\Build\vcvars64.bat" >nul)

echo [1/5] Installing git-hooks ^(cargo xtask install-hooks^)...
cargo xtask install-hooks
if %errorlevel% neq 0 (echo [WARNING] Failed to install hooks ^(the folder may not be a git repository^). Continuing... )

echo [2/5] Checking configuration file victory.toml...
if not exist "victory.toml" (
    if exist "victory.example.toml" (
        copy /Y "victory.example.toml" "victory.toml" >nul
        echo Created victory.toml from victory.example.toml.
        echo IMPORTANT: Make sure to set valid paths inside victory.toml!
    ) else (echo [WARNING] File victory.example.toml not found.)
) else (echo File victory.toml already exists, skipping copy.)

if "%SKIP_TESTS%"=="" echo [3/5] Running workspace tests ^(cargo test --workspace^)...
if "%SKIP_TESTS%"=="" cargo test --workspace
if "%SKIP_TESTS%"=="" if %errorlevel% neq 0 (echo [ERROR] Tests failed! & pause & exit /b 1)

if "%mode%"=="DEBUG" echo [4/5] Building DEBUG version... & cargo build --profile diag -p victory --features diagnostics & set "VIEXE=target\diag\victory.exe"
if "%mode%"==""      echo [4/5] Building release version ^(cargo build --release -p victory^)... & cargo build --release -p victory & set "VIEXE=target\release\victory.exe"
if %errorlevel% neq 0 (echo [ERROR] Release build failed! & pause & exit /b 1)

echo [5/5] Verifying configuration ^(%VIEXE% config^)...
target\release\victory.exe config
if %errorlevel% neq 0 (echo [WARNING] Configuration check returned an error. Check the paths in victory.toml. )

echo.
echo ========================================================
echo   Build completed successfully! Executable: %VIEXE%
echo   Example game launch: %VIEXE% run ^<path_to_game^>\^<game_exe^> --execute
echo ========================================================
pause
exit /b 0