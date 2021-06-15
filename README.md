# This is the repo for the paper: [Leveraging Two Types of Global Graph for Sequential Fashion Recommendation](https://arxiv.org/pdf/2105.07585.pdf)

## Requirements
1. OS: Ubuntu 16.04 or higher version
2. python3.7
3. Supported (tested) CUDA Versions: V10.2
4. python modules: refer to the modules in [requirements.txt](https://github.com/JoanDING/DGSR/blob/main/requirements.txt)


## Code Structure
1. The entry script for training and evaluation is: [train.py](https://github.com/JoanDING/DGSR/blob/main/train.py)
2. The config file is: [config.yaml](https://github.com/JoanDING/DGSR/blob/main/config.yaml)
3. The script for data preprocess and dataloader: [utility.py](https://github.com/JoanDING/DGSR/blob/main/utility.py)
4. The model folder: ./model/.
5. The experimental logs in tensorboard-format are saved in ./logs.
6. The experimental logs in txt-format are saved in ./performance.
7. The best model for each experimental setting is saved in ./model_saves.
8. The recommendation results in the evaluation are recorded in ./results.
9. The ./logs, ./performance, ./model_saves, ./results files will be generated automatically when first time runing the codes. 
10. The script [get_all_the_res.py](https://github.com/JoanDING/DGSR/blob/main/get_all_the_res.py)
 is used to print the performance of all the trained and tested models on the screen.


## How to Run
1. Download the [dataset](https://drive.google.com/file/d/1dFMu9-RvRa7a-47yYcN2VE5tPlYSxyD0/view?usp=sharing), decompress it and put it in the top directory with the following command. Note that the downloaded files include two datasets ulilized in the paper: iFashion and amazon_fashion.
    ```
     tar zxvf dgsr_dataset.tar.gz. 
    ```

2. Settings in the configure file config.yaml are basic experimental settings, which are usually fixed in the experiments. To tune other hyper-parameters, you can use command line to pass the parameters. The command line supported hyper-parameters including: the dataset (-d), sequence length (-l) and embedding size (-e). You can also specify which gpu device (-g) to use in the experiments. 

3. Run the training and evaluation with the specified hyper-parameters by the command: 
    ```
    python train.py -d=ifashion -l=5 -e=50 -g=0. 
    ```

4. During the training, you can monitor the training loss and the evaluation performance by Tensorboard. You can get into ./logs to track the curves of your training and evaluation with the following command:
    ```
    tensorboard --host="your host ip" --logdir=./
    ```

5. The performance of the model is saved in ./performance. You can get into the folder and check the detailed training process of any finished experiments (Compared with the tensorboard log save in ./logs, it is just the txt-version human-readable training log). To quickly check the results for all implemented experiments, you can also print the results of all experiments in a table format on the terminal screen by running: 
    ```
    python get_all_the_res.py
    ```

6. The best model will be saved in ./model_saves. 
