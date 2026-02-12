@echo off
REM Student-friendly MuJoCo environment starter for Windows
REM Works with WSL and Docker Desktop

setlocal enabledelayedexpansion

REM Configuration
set IMAGE_NAME=yguel/mujoco-desktop:v1.0
set CONTAINER_NAME=mujoco-novnc
set USE_LOCAL_ONLY=false
set SMALL_RAM_MODE=false
set CUSTOM_RAM_VALUE=
set VNC_RESOLUTION=1920x1080
set VNC_QUALITY=high
set NO_GPU_MODE=false
set DEBUG_MODE=false

REM Parse command line arguments
:parse_args
if "%~1"=="" goto end_parse
if /i "%~1"=="--local" (
    set USE_LOCAL_ONLY=true
    shift
    goto parse_args
)
if /i "%~1"=="--no_gpu" (
    set NO_GPU_MODE=true
    shift
    goto parse_args
)
if /i "%~1"=="--small_ram" (
    set SMALL_RAM_MODE=true
    shift
    goto parse_args
)
if /i "%~1"=="--ram" (
    set CUSTOM_RAM_VALUE=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--resolution" (
    set VNC_RESOLUTION=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--quality" (
    set VNC_QUALITY=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--debug" (
    set DEBUG_MODE=true
    shift
    goto parse_args
)
if /i "%~1"=="-h" goto show_help
if /i "%~1"=="--help" goto show_help
echo Unknown option: %~1
echo Use --help for usage information
exit /b 1

:show_help
echo Usage: %~nx0 [OPTIONS]
echo.
echo Options:
echo   --local           Use local Docker image only, skip remote version check
echo   --no_gpu          Force software rendering (disable GPU even if detected)
echo   --small_ram       Use conservative memory settings: min(2GB, 50%% RAM)
echo   --ram SIZE        Use specific memory amount (e.g., --ram 1g, --ram 512m)
echo   --resolution WxH  Set VNC resolution (default: 1920x1080)
echo   --quality LEVEL   Set VNC quality: high, medium, low (default: high)
echo   --debug           Enable debug mode with verbose output
echo   -h, --help        Show this help message
echo.
echo Examples:
echo   %~nx0                              # Normal mode: Full HD, high quality
echo   %~nx0 --resolution 1440x900       # Custom resolution
echo   %~nx0 --quality medium --small_ram # Medium quality, low memory
echo   %~nx0 --ram 512m --quality low    # Low resource usage
exit /b 0

:end_parse

echo.
echo ========================================
echo   MuJoCo Student Environment (Windows)
echo ========================================
echo Image: %IMAGE_NAME%
echo Container: %CONTAINER_NAME%
if "%USE_LOCAL_ONLY%"=="true" (
    echo Mode: LOCAL ONLY
) else (
    echo Mode: SMART UPDATE
)
echo.

REM Check if Docker is running
echo Checking Docker...
docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not running!
    echo Please start Docker Desktop and try again.
    exit /b 1
)
echo [OK] Docker is running

REM Setup workspace path
set WORKSPACE_PATH=%USERPROFILE%\rl\mujoco
echo.
echo Setting up workspace: %WORKSPACE_PATH%
if not exist "%WORKSPACE_PATH%\workspace" mkdir "%WORKSPACE_PATH%\workspace"
if not exist "%WORKSPACE_PATH%\workspace\notebooks" mkdir "%WORKSPACE_PATH%\workspace\notebooks"
if not exist "%WORKSPACE_PATH%\workspace\examples" mkdir "%WORKSPACE_PATH%\workspace\examples"
if not exist "%WORKSPACE_PATH%\workspace\models" mkdir "%WORKSPACE_PATH%\workspace\models"

REM Calculate memory settings
if not "%CUSTOM_RAM_VALUE%"=="" (
    set SHM_SIZE=%CUSTOM_RAM_VALUE%
    echo Memory: %CUSTOM_RAM_VALUE% (custom)
) else (
    if "%SMALL_RAM_MODE%"=="true" (
        set SHM_SIZE=2g
        echo Memory: 2GB (small RAM mode)
    ) else (
        set SHM_SIZE=4g
        echo Memory: 4GB (default)
    )
)

REM Set VNC quality parameters
if /i "%VNC_QUALITY%"=="high" (
    set VNC_DEPTH=24
    set VNC_DPI=96
) else if /i "%VNC_QUALITY%"=="medium" (
    set VNC_DEPTH=16
    set VNC_DPI=96
) else if /i "%VNC_QUALITY%"=="low" (
    set VNC_DEPTH=8
    set VNC_DPI=72
) else (
    echo Warning: Unknown quality '%VNC_QUALITY%', using high
    set VNC_DEPTH=24
    set VNC_DPI=96
)

REM Stop existing container if running
echo.
echo Checking for existing container...
docker ps -a | findstr /C:"%CONTAINER_NAME%" >nul 2>&1
if not errorlevel 1 (
    echo Stopping and removing existing container...
    docker stop %CONTAINER_NAME% >nul 2>&1
    docker rm %CONTAINER_NAME% >nul 2>&1
)

REM Check for local image
echo.
echo Checking image availability...
docker images --format "{{.Repository}}:{{.Tag}}" | findstr /C:"%IMAGE_NAME%" >nul 2>&1
if not errorlevel 1 (
    echo [OK] Local image found: %IMAGE_NAME%
    if "%USE_LOCAL_ONLY%"=="false" (
        echo Checking for updates...
        docker pull %IMAGE_NAME% >nul 2>&1
        if not errorlevel 1 (
            echo [OK] Image updated
        ) else (
            echo [INFO] Using local image
        )
    )
) else (
    if "%USE_LOCAL_ONLY%"=="true" (
        echo [ERROR] No local image found and --local flag prevents pulling
        exit /b 1
    )
    echo Pulling image from Docker Hub...
    docker pull %IMAGE_NAME%
    if errorlevel 1 (
        echo [ERROR] Failed to pull image
        exit /b 1
    )
)

REM GPU detection (Windows with Docker Desktop uses WSL2 backend)
echo.
echo Detecting graphics capabilities...
set GPU_FLAGS=
if "%NO_GPU_MODE%"=="true" (
    echo [INFO] GPU disabled by --no_gpu flag
    set GPU_ENV=-e LIBGL_ALWAYS_SOFTWARE=1 -e MUJOCO_GL=osmesa
) else (
    REM Check if NVIDIA GPU is available via nvidia-smi
    nvidia-smi >nul 2>&1
    if not errorlevel 1 (
        echo [OK] NVIDIA GPU detected
        REM Test if Docker supports --gpus
        docker run --rm --gpus all alpine:latest echo test >nul 2>&1
        if not errorlevel 1 (
            echo [OK] GPU acceleration enabled
            set GPU_FLAGS=--gpus all
            set GPU_ENV=-e NVIDIA_VISIBLE_DEVICES=all
        ) else (
            echo [WARNING] GPU detected but Docker GPU support not configured
            echo [INFO] Using software rendering
            set GPU_ENV=-e LIBGL_ALWAYS_SOFTWARE=1
        )
    ) else (
        echo [INFO] No NVIDIA GPU detected, using software rendering
        set GPU_ENV=-e LIBGL_ALWAYS_SOFTWARE=1
    )
)

REM Convert Windows path to Docker-compatible path
set DOCKER_WORKSPACE_PATH=%WORKSPACE_PATH:\=/%
set DOCKER_WORKSPACE_PATH=%DOCKER_WORKSPACE_PATH:C:=/c%
set DOCKER_WORKSPACE_PATH=%DOCKER_WORKSPACE_PATH:D:=/d%
set DOCKER_WORKSPACE_PATH=%DOCKER_WORKSPACE_PATH:E:=/e%

echo.
echo VNC Display Configuration:
echo   Resolution: %VNC_RESOLUTION%
echo   Quality: %VNC_QUALITY%
echo   Color depth: %VNC_DEPTH%-bit
echo   DPI: %VNC_DPI%
echo.
echo Starting container...
echo.
echo Access URLs:
echo   Desktop: http://localhost:6080
echo   Jupyter: http://localhost:8888
echo.
echo Workspace: %WORKSPACE_PATH%
echo.
echo Press Ctrl+C to stop the environment
echo ==========================================
echo.

REM Build and run Docker command
docker run -it --rm ^
    --name %CONTAINER_NAME% ^
    --shm-size=%SHM_SIZE% ^
    -p 6080:6080 ^
    -p 8888:8888 ^
    -v "%DOCKER_WORKSPACE_PATH%/workspace:/home/student/workspace" ^
    -e VNC_RESOLUTION=%VNC_RESOLUTION% ^
    -e VNC_DEPTH=%VNC_DEPTH% ^
    -e VNC_DPI=%VNC_DPI% ^
    -e NOVNC_PORT=6080 ^
    -e VNC_PORT=5901 ^
    -e DISPLAY=:1 ^
    -e HOST_UID=1000 ^
    -e HOST_GID=1000 ^
    -e JUPYTER_ALLOW_ROOT=yes ^
    %GPU_FLAGS% ^
    %GPU_ENV% ^
    %IMAGE_NAME%

echo.
echo Environment stopped.
