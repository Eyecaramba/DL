Q.torch.nn.Identity가 존재하는 이유? 사용목적?  
A. [purpose of Identity](https://stackoverflow.com/questions/64229717/what-is-the-idea-behind-using-nn-identity-for-residual-learning)

조건에 따라 layer를 발동시키지 말아야할 경우 if else 문에 torch.nn.Identity를 사용하여 모델을 구성할 수 있다. 이렇게 되면 nn.Sequential Module을 쉽게 사용할 수 있다. 


why is teh super constructor necessary in Pytorch custom Module?
(https://stackoverflow.com/questions/63058355/why-is-the-super-constructor-necessary-in-pytorch-custom-modules)

super().__init__ 의 역할은 parent module 로부터 attribute를 상속받아야 하기 때문이다.  
python에서는 super()명령어를 통해 명시적으로 선얺을 해야한다.  



Modle이 선언될 때 안에 있는 Module이 초기화가 되고 model에 값을 넣으면 forward가 진행되면서 Module이 작동한다.  


Q. What is the difference between registed_buffer and register_parameter of nn.Module?  
A. https://discuss.pytorch.org/t/what-is-the-difference-between-register-buffer-and-register-parameter-of-nn-module/32723


Q. Buffer의 용도는 무엇인가요?
A. 


Q. Hook의 용도는?  Verbose Execution, Feature Extraction, Gradient Clipping 
A1. https://medium.com/the-dl/how-to-use-pytorch-hooks-5041d777f904  
A2. https://blog.paperspace.com/pytorch-hooks-gradient-clipping-debugging/
A3. 


Q. 이미 존재하는 python object에 attribute를 추가하는 방법  
A. 



Pytorch Lightning  Multi GPU   
https://lightning.ai/docs/pytorch/stable/accelerators/gpu_basic.html


Pytorch and Ray  
https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html

DDP Tutorial  
https://pytorch.org/tutorials/intermediate/ddp_tutorial.html



GPU 에러 정리  
https://brstar96.github.io/devlog/shoveling/2020-01-03-device_error_summary/  

Pytorch에서 자주 발생하는 에러  
https://pytorch.org/docs/stable/notes/faq.html

 OOM에서 GPU 메모리 flush하기  
 https://discuss.pytorch.org/t/how-to-clean-gpu-memory-after-a-runtimeerror/28781  


dataloader에서 sampler의 다양한 방법   
https://towardsdatascience.com/pytorch-basics-sampling-samplers-2a0f29f0bf2a 



ray와 pytorch 함께 사용하기 
https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html

Ray -
  
Multi node Multi Processing 지원 모듈    
ML/DL 병렬 처리를 위해 개발된 모듈   
기본적으로 현재의 분산 병렬 ML/DL 모듈의 표준  
hyperparameter Search를 위한 다양한 모듈 제공함  

CPU/GPU가 여러개인 상태에서 hyper parameter search를 위해 사용된다.  

1. 하이퍼 파라미터를 탐색할 범위를 지정해준다. config에 설정해준다.  
2. ASHA scheduler : 가망이 없어보이는 것들은 먼저 걸러낸다.  
baysian 
3. 안쓰는 데이터들을 끝까지 탐색하는 것이 낭비이기 때문이다.  
4. Tune.run : 여러개의 GPU에 뿌려주고 학습을 시작하게 된다.  

ray 사용시 주의할 점 : 
train에 관한 전반적인 과정이 하나의 함수로 들어가 있어야 한다.  
병렬처리하는 파이썬 코드를 보면 쉽게 알 수 있다.  

하이퍼파라미터 튜닝은 마지막에 끝날때까지 끝난게 아닐 때 시도 
