#!/bin/bash
#SBATCH --job-name=download-datasets
#SBATCH --time=12:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_scripts/logs/download-datasets-%j.out
#SBATCH --error=slurm_scripts/logs/download-datasets-%j.err

echo "=========================================="
echo "BigANN Dataset Download Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 200G"
echo "Start time: $(date)"
echo "=========================================="

# Load environment
module purge
module load Python/3.11.3-GCCcore-12.3.0

# Activate virtual environment
source $HOME/code/flatnav/venv/bin/activate

# Go to flatnav directory
cd $HOME/code/flatnav

# Download ground truth files if not already present
echo ""
echo "=========================================="
echo "Step 1: Downloading Ground Truth Files"
echo "=========================================="

if [ -d "GT_10M" ]; then
    echo "✓ GT_10M directory already exists, skipping download"
else
    echo "Downloading GT_10M..."
    wget https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/GT_10M_v2.tgz
    echo "Extracting GT_10M..."
    tar -xzvf GT_10M_v2.tgz
    echo "✓ GT_10M downloaded and extracted"
fi

if [ -d "GT_100M" ]; then
    echo "✓ GT_100M directory already exists, skipping download"
else
    echo "Downloading GT_100M..."
    wget https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/GT_100M_v2.tgz
    echo "Extracting GT_100M..."
    tar -xzvf GT_100M_v2.tgz
    echo "✓ GT_100M downloaded and extracted"
fi

# ============================================
# Download and convert text2image dataset
# ============================================
echo ""
echo "=========================================="
echo "Step 2: Text2Image Dataset"
echo "=========================================="

DATASET="text2image"
mkdir -p data/${DATASET}

if [ -f "data/${DATASET}/tti_base.1B.fbin" ]; then
    echo "✓ Base file already exists, skipping download"
else
    echo "Downloading text2image base dataset (1B vectors)..."
    echo "Start time: $(date)"
    wget -O tti_base.1B.fbin https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/base.1B.fbin

    if [ $? -eq 0 ]; then
        echo "✓ Download complete"
        mv tti_base.1B.fbin data/${DATASET}/tti_base.1B.fbin
        ls -lh data/${DATASET}/tti_base.1B.fbin
    else
        echo "✗ Download failed"
        exit 1
    fi
fi

if [ -f "data/${DATASET}/tti_query.learn.50M.fbin" ]; then
    echo "✓ Query file already exists, skipping download"
else
    echo "Downloading text2image query dataset (50M vectors)..."
    echo "Start time: $(date)"
    wget -O tti_query.learn.50M.fbin https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.learn.50M.fbin

    if [ $? -eq 0 ]; then
        echo "✓ Download complete"
        mv tti_query.learn.50M.fbin data/${DATASET}/tti_query.learn.50M.fbin
        ls -lh data/${DATASET}/tti_query.learn.50M.fbin
    else
        echo "✗ Download failed"
        exit 1
    fi
fi

echo ""
echo "Converting text2image query dataset..."
python convert_bigann_datasets.py data/${DATASET}/tti_query.learn.50M.fbin queries
if [ $? -ne 0 ]; then
    echo "✗ Query conversion failed"
    exit 1
fi

echo ""
echo "Converting text2image base dataset..."
python convert_bigann_datasets.py data/${DATASET}/tti_base.1B.fbin train
if [ $? -ne 0 ]; then
    echo "✗ Base conversion failed"
    exit 1
fi

echo "✓ Text2Image dataset complete"
ls -lh data/${DATASET}/*.npy

# ============================================
# Download and convert msspacev dataset
# ============================================
echo ""
echo "=========================================="
echo "Step 3: MSSpaceV Dataset"
echo "=========================================="

DATASET="msspacev"
mkdir -p data/${DATASET}

if [ -d "SPTAG" ]; then
    echo "✓ SPTAG repository already exists, skipping clone"
else
    echo "Cloning SPTAG repository (this may take a while)..."
    echo "Start time: $(date)"
    export GIT_LFS_SKIP_SMUDGE=1
    git clone --recurse-submodules https://github.com/microsoft/SPTAG

    if [ $? -ne 0 ]; then
        echo "✗ Git clone failed"
        exit 1
    fi
    echo "✓ Repository cloned"
fi

if [ -f "data/${DATASET}/vectors.bin" ]; then
    echo "✓ Vectors file already exists, skipping move"
else
    echo "Moving vectors.bin..."
    if [ -d "SPTAG/datasets/SPACEV1B/vectors.bin" ]; then
        mv SPTAG/datasets/SPACEV1B/vectors.bin/ data/${DATASET}/
        echo "✓ Vectors moved"
        ls -lh data/${DATASET}/vectors.bin
    else
        echo "✗ SPTAG/datasets/SPACEV1B/vectors.bin not found"
        echo "Contents of SPTAG/datasets/:"
        ls -lR SPTAG/datasets/ 2>/dev/null || echo "Directory not found"
        exit 1
    fi
fi

if [ -f "data/${DATASET}/query.bin" ]; then
    echo "✓ Query file already exists, skipping move"
else
    echo "Moving query.bin..."
    if [ -f "SPTAG/datasets/SPACEV1B/query.bin" ]; then
        mv SPTAG/datasets/SPACEV1B/query.bin data/${DATASET}/
        echo "✓ Query moved"
        ls -lh data/${DATASET}/query.bin
    else
        echo "✗ SPTAG/datasets/SPACEV1B/query.bin not found"
        exit 1
    fi
fi

echo ""
echo "Converting msspacev query dataset..."
python convert_spacev_dataset.py data/${DATASET}/query.bin queries
if [ $? -ne 0 ]; then
    echo "✗ Query conversion failed"
    exit 1
fi

echo ""
echo "Converting msspacev base dataset..."
python convert_spacev_dataset.py data/${DATASET}/vectors.bin train
if [ $? -ne 0 ]; then
    echo "✗ Base conversion failed"
    exit 1
fi

echo "✓ MSSpaceV dataset complete"
ls -lh data/${DATASET}/*.npy

# ============================================
# Convert ground truth files
# ============================================
echo ""
echo "=========================================="
echo "Step 4: Converting Ground Truth Files"
echo "=========================================="

for DATASET in text2image msspacev; do
    echo ""
    echo "Converting ground truth for ${DATASET}..."

    if [ -f "GT_10M/${DATASET}-10M" ]; then
        python convert_bigann_datasets.py GT_10M/${DATASET}-10M gt
        echo "✓ GT_10M converted for ${DATASET}"
    else
        echo "⚠ GT_10M/${DATASET}-10M not found, skipping"
    fi

    if [ -f "GT_100M/${DATASET}-100M" ]; then
        python convert_bigann_datasets.py GT_100M/${DATASET}-100M gt
        echo "✓ GT_100M converted for ${DATASET}"
    else
        echo "⚠ GT_100M/${DATASET}-100M not found, skipping"
    fi
done

# ============================================
# Final summary
# ============================================
echo ""
echo "=========================================="
echo "✓ All datasets downloaded and converted!"
echo "=========================================="
echo "End time: $(date)"
echo ""
echo "Downloaded datasets:"
ls -lh data/text2image/*.npy 2>/dev/null && echo "" || echo "⚠ text2image files not found"
ls -lh data/msspacev/*.npy 2>/dev/null || echo "⚠ msspacev files not found"

exit 0
