# **Multi-hop Knowledge Base Question Answering with an Iterative Sequence Matching Model**
This is the code for paper:\
**[Multi-hop Knowledge Base Question Answering with an Iterative Sequence Matching Model]**()\
Yunshi Lan, Shuohang Wang, Jing Jiang\
Will appear at [ICDM 2019](http://icdm2019.bigke.org/)\

If you find this code useful in your research, please cite
>@inproceedings{lan:icdm19,\
>title={Multi-hop Knowledge Base Question Answering with an Iterative Sequence Matching Model},\
>author={Lan, Yunshi and Wang, Shuohang and Jiang, Jing},\
>booktitle={Proceedings of the IEEE International Conference on Data Mining (ICDM)},\
>year={2019}\
>}

## **Setups** 
All code was developed and tested in the following environment. \
-Ubuntu 16.04\
-Python 3.7.1\
-Pytorch 1.1.0\
Download the code and data:
```
git clone https://github.com/lanyunshi/Multi-hopQA.git
```
## **Running Pre-trained Models**
```
python code/Multi-hopKBQA_runner.py --task 1 # Run pre-trained datasets: {1, 2, 3}
```
A full list of commands can be found in ```code/options.py```. The training script has a number of command-line flags that you can use to configure the model architecture, hyper-parameters and input/output settings. You can change the saved arguments in ```code/Multi-hopKBQA_runner.py``` .
## **Running a New Model**
```
python code/Multi-hopKBQA_runner.py --task 0
```
Task 0 is set to train your own model. The data is pre-processed and the model is initialized randomly. To train a new model, a new folder will be generated in the ```trained_model/```. A general good argument setting is:
- learning_rate : 0.0001\
- hidden_dimension : 200 \
- dropout : 0.0\
- max_epochs : 20\
- threshold : 0.5\
- max_hop : 3\
- top : 3\
## **Performance**
We re-implemented the model since the cleaning up of the code leads to some variable inconsistence of the previously saved model. The number is slightly different from the ones reported in the paper. The macro average F1 score is shown as follows:\
```
|Dataset|MetaQA|PathQuestion|WC2014|
|---|---|---|---|
|Dev|98.7|98.0|99.9|
|Test|98.8|96.2|99.9|
```
