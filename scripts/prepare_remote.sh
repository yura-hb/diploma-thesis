
apt update
#apt upgrade -y

apt-get install python -y
apt-get install curl openssl -y

snap install gh

mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh


~/miniconda3/bin/conda init bash

exec bash


eval `ssh-agent`

# ssh-add

#conda create --name main python=3.11
