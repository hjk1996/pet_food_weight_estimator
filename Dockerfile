FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime

RUN pip install timm==0.6.11 \
                scikit-learn==1.0.2 \
                tensorboard==2.10.1 \
                matplotlib==3.6.2 \
                opencv-python==4.6.0.66 \
                Pillow==9.3.0 \
                PyYAML==6.0 \
                requests==2.28.1 \
                scipy==1.9.3 \ 
                tqdm==4.64.1 \
                protobuf==3.19.6 \
                pandas==1.5.2 \ 
                seaborn==0.12.1 \ 
                thop \ 
                ipython \ 
                psutil 

RUN apt-get -y update
RUN apt-get install -y zip htop screen libgl1-mesa-glx git
# RUN git clone https://github.com/hjk1996/pet_food_weight_estimator.git
# RUN git clone https://github.com/hjk1996/yolov7.git

ENV PYTHONPATH="$PYTHONPATH:/workspace/yolov7"



