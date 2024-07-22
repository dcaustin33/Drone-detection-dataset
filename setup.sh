pip install poetry;
poetry install;
cd ../;
git clone https://github.com/dcaustin33/GroundingDINO;
cd GroundingDINO;
git checkout feat/batch;
bash setup.sh;
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth;
cd ../Drone-detection-dataset;