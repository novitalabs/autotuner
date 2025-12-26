#!/bin/bash
# Install inference-autotuner on a remote machine for worker deployment
# This script is designed to be run on the target machine

set -e

INSTALL_PATH="${INSTALL_PATH:-/opt/inference-autotuner}"
PYTHON_VERSION="${PYTHON_VERSION:-python3}"

echo "========================================"
echo "Inference Autotuner Worker Installation"
echo "========================================"
echo ""
echo "Install path: $INSTALL_PATH"
echo ""

# Check if running as root or with sudo
if [ "$EUID" -ne 0 ]; then
    echo "This script requires root privileges."
    echo "Please run with: sudo $0"
    exit 1
fi

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
    OS_VERSION=$VERSION_ID
else
    OS="unknown"
fi

echo "Detected OS: $OS $OS_VERSION"
echo ""

# Install system dependencies
echo "[1/5] Installing system dependencies..."
case $OS in
    ubuntu|debian)
        apt-get update -qq
        apt-get install -y -qq python3 python3-pip python3-venv git netcat-openbsd curl > /dev/null
        ;;
    centos|rhel|fedora)
        if command -v dnf &> /dev/null; then
            dnf install -y -q python3 python3-pip git nc curl
        else
            yum install -y -q python3 python3-pip git nc curl
        fi
        ;;
    *)
        echo "Warning: Unknown OS. Please ensure python3, pip, git are installed."
        ;;
esac
echo "✓ System dependencies installed"

# Check Python version
PYTHON_CMD=""
for cmd in python3.10 python3.11 python3.9 python3; do
    if command -v $cmd &> /dev/null; then
        version=$($cmd -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        major=$(echo $version | cut -d. -f1)
        minor=$(echo $version | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 9 ]; then
            PYTHON_CMD=$cmd
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "Error: Python 3.9+ is required but not found."
    exit 1
fi
echo "✓ Using Python: $PYTHON_CMD ($($PYTHON_CMD --version))"

# Create install directory
echo ""
echo "[2/5] Setting up install directory..."
mkdir -p "$INSTALL_PATH"
mkdir -p "$INSTALL_PATH/logs"
echo "✓ Created $INSTALL_PATH"

# Check if project files exist (will be synced by deploy_worker)
if [ ! -f "$INSTALL_PATH/requirements.txt" ]; then
    echo ""
    echo "Note: Project files not found. They will be synced by the deploy_worker tool."
    echo "      If installing manually, copy the project files to $INSTALL_PATH"
fi

# Create virtual environment
echo ""
echo "[3/5] Creating virtual environment..."
if [ ! -d "$INSTALL_PATH/env" ]; then
    $PYTHON_CMD -m venv "$INSTALL_PATH/env"
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Install Python dependencies
echo ""
echo "[4/5] Installing Python dependencies..."
if [ -f "$INSTALL_PATH/requirements.txt" ]; then
    "$INSTALL_PATH/env/bin/pip" install --upgrade pip -q
    "$INSTALL_PATH/env/bin/pip" install -r "$INSTALL_PATH/requirements.txt" -q
    echo "✓ Python dependencies installed"
else
    echo "⚠ requirements.txt not found. Dependencies will be installed after file sync."
fi

# Check NVIDIA driver
echo ""
echo "[5/5] Checking GPU setup..."
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo "✓ NVIDIA driver found"
    echo "  GPUs: $GPU_COUNT x $GPU_NAME"
else
    echo "⚠ nvidia-smi not found. GPU support may be limited."
fi

# Final status
echo ""
echo "========================================"
echo "Installation Complete!"
echo "========================================"
echo ""
echo "Install path: $INSTALL_PATH"
echo "Python: $($INSTALL_PATH/env/bin/python --version 2>/dev/null || echo 'venv pending')"
echo ""
echo "Next steps:"
echo "  1. Sync project files to $INSTALL_PATH"
echo "  2. Configure .env.worker with Redis connection"
echo "  3. Run: $INSTALL_PATH/scripts/start_remote_worker.sh"
echo ""
