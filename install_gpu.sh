git submodule add https://github.com/tensorflow/models.git models
cd models/research
# Compile protos.
protoc object_detection/protos/*.proto --python_out=.
# Install TensorFlow Object Detection API.
cp object_detection/packages/tf2/setup.py .
python -m pip install --use-feature=2020-resolver .
cd ../..
sed 's/tensorflow=/tensorflow-gpu=/' requirements.txt > new_requirements.txt
pip install -r new_requirements.txt
rm -rf new_requirements.txt