conda create -n macro_place python=3.8
conda activate macro_place
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install hydra-core --upgrade
pip install tensorboardX
pip install tensorboard
pip install gym
pip install treelib
pip install matplotlib