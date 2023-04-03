*INTRO*   
이미지 분류 모델 정리 

<br>
<br>
6
__목차__  
1. 


## 1. Introduction
___

딥러닝 모델의 깊이가 깊어질수록 더 학습시키기 어려워졌다.  
20-layer보다 56-layer의 __training error__ 가 더 높은 것을 통해 확인할 수 있다.  

저자는 F(x)라는 함수에 바로 근사하는 것 보다. H(x) = F(x) + x 함수를 만들어 H(x)를 학습시킨 뒤 F(x) = H(x) - x로 바꾸어 F(x)함수를 간접적으로 찾는 방법이 더 쉽다고 가정한다. 

어떤 layer가 항등함수가 되도록 학습시키는 것은 어렵지만 layer의 모든 

딥러닝 모델이 이렇게 학습시키게 하기 위해 shortcut 구조를 만든다.  




## 2. Related Work 
___
1) Residual Representation 

2) Shortcut Connection 

## 3. Deep Residual Learning 
___
1) Residual Learning  

항등함수가 최적의 함수인 경우 이 함수를 근사하는 것은 매우 어렵다.  
그러나 skip connection으로 항등함수가 연결되어 있는 경우 model은 layer의 weight를 0으로 만들면 되기 때문에 쉽게 항등함수를 구현할 수 있다. 즉 같은 함수를 근사한다고 하더라도 학습 난이도는 차이가 있다.  

2) Identity Mapping by Shortcuts 

3) Network Architectures 




## 4. CNN 발전 과정 
___

1. AlexNet
2. VGGNet
3. GoogLeNet
4. ResNet



## Ref
---
https://dennybritz.com/posts/deep-learning-ideas-that-stood-the-test-of-time/ 