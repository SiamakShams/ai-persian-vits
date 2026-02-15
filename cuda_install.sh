# 1. Remove old CUDA
sudo apt-get --purge remove '*nvidia*' '*cuda*'

# 2. Install NVIDIA driver (560+)
sudo apt-get update
sudo apt-get install -y nvidia-driver-560

# 3. Install CUDA 12.1
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-ubuntu2204-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo apt-key adv --fetch-keys /var/cuda-repo-ubuntu2204-12-1-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda-12-1

# 4. Install cuDNN 9.0
# Download from: https://developer.nvidia.com/cudnn
# Follow: https://docs.nvidia.com/deeplearning/cudnn/install-guide/

# 5. Verify
nvidia-smi
nvcc --version
