# Sparse Adversarial Perturbations for Videos
This is the official code for the paper "Sparse Adversarial Perturbations for Videos" accepted by AAAI2019.

The code is tested on the tensorfow >= 1.3.0

Please dowload the checkpoints from https://drive.google.com/file/d/1i-16KN9JEn4pvdvSKZt-J1HtwzQRE8xm/view?usp=sharing

To run the code, please use the following command:

python l21_optimization.py -i video_data -o output/Inception2 --model Inception2 --file_list video_data/batch_test/test.csv

The generated adversarial videos will be stored in the folder "output"

