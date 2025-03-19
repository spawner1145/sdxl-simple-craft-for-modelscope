#!/bin/sh

# 可选调试模式，通过环境变量 DEBUG=true 启用
if [ "$DEBUG" = "true" ]; then
    set -x
fi

# 备份原始 DNS 配置
ORIGINAL_RESOLV=$(mktemp)
if [ -f /etc/resolv.conf ]; then
    cp /etc/resolv.conf "$ORIGINAL_RESOLV" || { echo "Failed to backup resolv.conf" >&2; exit 1; }
fi

# 设置阿里云和腾讯云公共 DNS
echo "Setting DNS to Aliyun (223.5.5.5) and Tencent (119.29.29.29)..." >&2
echo "nameserver 223.5.5.5" | tee /etc/resolv.conf > /dev/null
echo "nameserver 119.29.29.29" | tee -a /etc/resolv.conf > /dev/null

# 清理函数，恢复原始 DNS 配置
cleanup() {
    echo "Restoring original DNS configuration..." >&2
    if [ -f "$ORIGINAL_RESOLV" ]; then
        cp "$ORIGINAL_RESOLV" /etc/resolv.conf || echo "Failed to restore resolv.conf" >&2
        rm -f "$ORIGINAL_RESOLV"
    fi
}

# 注册清理函数，脚本退出时执行
trap cleanup EXIT

# 检查是否安装了 aria2，如果没有，则尝试安装
if ! command -v aria2c > /dev/null 2>&1; then
    echo "aria2c could not be found, attempting to install..." >&2
    export DEBIAN_FRONTEND=noninteractive

    # 尝试使用默认源安装
    if ! sudo apt-get update; then
        echo "apt-get update failed, switching to official Ubuntu source..." >&2
        # 备份原始源
        sudo cp /etc/apt/sources.list /etc/apt/sources.list.bak
        # 更换为官方 Ubuntu 源（以 Ubuntu 22.04 为例）
        echo "deb http://archive.ubuntu.com/ubuntu/ jammy main restricted universe multiverse" | sudo tee /etc/apt/sources.list
        echo "deb http://archive.ubuntu.com/ubuntu/ jammy-updates main restricted universe multiverse" | sudo tee -a /etc/apt/sources.list
        echo "deb http://archive.ubuntu.com/ubuntu/ jammy-security main restricted universe multiverse" | sudo tee -a /etc/apt/sources.list
        # 再次尝试更新
        if ! sudo apt-get update; then
            echo "apt-get update failed even after switching sources." >&2
            exit 1
        fi
    fi

    # 安装 aria2
    if ! sudo apt-get install -y aria2; then
        echo "apt-get install aria2 failed." >&2
        exit 1
    fi
fi

# 参数检查：需要 URL 和输出目录
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <url> <output_dir>" >&2
    exit 1
fi

INPUT=$1
OUTPUT_DIR=$2

# 确保目标目录存在
mkdir -p "$OUTPUT_DIR" || { echo "Failed to create directory $OUTPUT_DIR" >&2; exit 1; }

# 检查输入是否包含 @，如果是，则分离名字和网址
if echo "$INPUT" | grep -q "@"; then
    CUSTOM_NAME=$(echo "$INPUT" | cut -d'@' -f1)
    URL=$(echo "$INPUT" | cut -d'@' -f2-)
    OUTPUT_PATH="$OUTPUT_DIR/$CUSTOM_NAME"
    
    # 检查文件是否已存在
    if [ -f "$OUTPUT_PATH" ]; then
        echo "File $OUTPUT_PATH already exists, skipping download." >&2
        exit 0
    fi
    
    echo "Starting download from $URL to $OUTPUT_PATH with custom name..." >&2
    aria2c -x 16 -s 16 -c -k 1M -o "$CUSTOM_NAME" "$URL" -d "$OUTPUT_DIR" --summary-interval=60 > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "aria2c download failed." >&2
        exit 1
    fi
else
    URL="$INPUT"
    OUTPUT_FILE=$(basename "$URL")
    OUTPUT_PATH="$OUTPUT_DIR/$OUTPUT_FILE"
    
    # 检查文件是否已存在
    if [ -f "$OUTPUT_PATH" ]; then
        echo "File $OUTPUT_PATH already exists, skipping download." >&2
        exit 0
    fi
    
    echo "Starting download from $URL to $OUTPUT_PATH with default name..." >&2
    aria2c -x 16 -s 16 -c -k 1M "$URL" -d "$OUTPUT_DIR" --summary-interval=60 > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "aria2c download failed." >&2
        exit 1
    fi
fi

echo "Download completed successfully." >&2