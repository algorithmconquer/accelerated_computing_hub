#!/bin/bash
# Day 2 快速开始脚本
# 自动编译和运行所有示例程序

echo "========================================"
echo "Day 2: CUDA编程基础深入"
echo "快速开始脚本"
echo "========================================"
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查CUDA环境
echo -e "${YELLOW}[1/5] 检查CUDA环境...${NC}"
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}错误: nvcc未找到，请检查CUDA安装${NC}"
    exit 1
fi
echo -e "${GREEN}✓ CUDA已安装: $(nvcc --version | grep release | awk '{print $5}')${NC}"
echo ""

# 检查GPU
echo -e "${YELLOW}[2/5] 检查GPU设备...${NC}"
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}错误: nvidia-smi未找到，请检查GPU驱动${NC}"
    exit 1
fi
echo -e "${GREEN}✓ GPU可用:$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)${NC}"
echo ""

# 编译程序
echo -e "${YELLOW}[3/5] 编译CUDA程序...${NC}"

echo "  编译 thread_organization..."
if nvcc -O3 -o thread_organization thread_organization.cu 2>&1; then
    echo -e "  ${GREEN}✓ 编译成功${NC}"
else
    echo -e "  ${RED}✗ 编译失败${NC}"
    exit 1
fi

echo "  编译 shared_memory_demo..."
if nvcc -O3 -o shared_memory_demo shared_memory_demo.cu 2>&1; then
    echo -e "  ${GREEN}✓ 编译成功${NC}"
else
    echo -e "  ${RED}✗ 编译失败${NC}"
    exit 1
fi

echo "  编译 matrix_multiplication..."
if nvcc -O3 -o matrix_multiplication matrix_multiplication.cu 2>&1; then
    echo -e "  ${GREEN}✓ 编译成功${NC}"
else
    echo -e "  ${RED}✗ 编译失败${NC}"
    exit 1
fi
echo ""

# 检查Python环境
echo -e "${YELLOW}[4/5] 检查Python环境...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
else
    echo -e "${RED}错误: Python未找到${NC}"
    exit 1
fi

if $PYTHON_CMD -c "import torch" 2>&1; then
    echo -e "${GREEN}✓ PyTorch已安装${NC}"
else
    echo -e "${YELLOW}警告: PyTorch未安装，将跳过Python验证脚本${NC}"
    PYTHON_AVAILABLE=false
fi
echo ""

# 运行程序
echo -e "${YELLOW}[5/5] 运行示例程序...${NC}"
echo ""
echo "========================================"
echo "示例1: 线程组织可视化"
echo "========================================"
./thread_organization
echo ""
read -p "按Enter继续下一个示例..."

echo ""
echo "========================================"
echo "示例2: 共享内存示例"
echo "========================================"
./shared_memory_demo
echo ""
read -p "按Enter继续下一个示例..."

echo ""
echo "========================================"
echo "示例3: 矩阵乘法优化"
echo "========================================"
./matrix_multiplication
echo ""

if [ "$PYTHON_AVAILABLE" != false ]; then
    echo ""
    echo "========================================"
    echo "示例4: GPU性能验证 (Python)"
    echo "========================================"
    $PYTHON_CMD verify_gpu_performance.py
    echo ""
fi

echo "========================================"
echo -e "${GREEN}所有示例运行完成!${NC}"
echo "========================================"
echo ""
echo "您可以随时重新运行这些程序:"
echo "  ./thread_organization"
echo "  ./shared_memory_demo"
echo "  ./matrix_multiplication"
echo "  python verify_gpu_performance.py"
echo ""
echo "查看README.md了解更多详情。"
