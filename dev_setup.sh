cd ..
git clone github@github.com:565353780/base-trainer.git
git clone github@github.com:565353780/camera-control.git
git clone --depth 1 https://github.com/camenduru/simple-knn.git
git clone --depth 1 https://github.com/rahul-goel/fused-ssim.git

cd base-trainer
./dev_setup.sh

cd ../camera-control
./dev_setup.sh

cd ../simple-knn
python setup.py install

cd ../fused-ssim
python setup.py install

pip install plyfile
