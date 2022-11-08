FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime

RUN pip install timm==0.6.11 \
                pandas==1.3.5 \ 
                scikit-learn==1.0.2 \
                tensorboard==2.10.1

