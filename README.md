# MNIST-Classification-from-Local-Data </br>
_____________________________________________________________
This project is an implementation of classification of MNIST handwritten numbers data taken from a local matlab file,
with the help of Pytorch. </br>
The Convolutional Neural Network (CNN) is composed of the following: </br>

- an input layer (784 nodes)
- first convolution layer
- first max-pooling layer
- second convolution layer
- second max-pooling layer
- third fully-connected layer
- output layer (10 nodes)


</br>

> 본 프로젝트는 Pytorch를 이용하지만 torch에서 사용 및 다운로드를 지원하는 MNIST 파일이 아닌 컴퓨터 로컬 디렉토리에 저장되어 있는 MNIST.mat 형식의 파일을
  불러와 텐서 형식으로 불러온 후, 학습 및 테스트하는 프로그램입니다. 
  
</br>
이 합성곱 신경망으로 구현된 코드는 다음과 같이 구성됩니다:</br>

- 입력층
- 1번 합성곱층
- 1번 풀링층
- 2번 합성곱층
- 2번 합성곱층
- 3번 완전연결층
- 출력층

</br>
</br>
The main purpose of this personal project was firstly to get familiar with Pytorch and Neural Networks, and also to try to
understand the details of the data in detail; this was why the project was conducted with a local MNIST.mat file. 
</br>
Pytorch and Tensorflow programs provide convenient libraries to download and use data such as MNIST easily. However, I wanted
to try and load a matlab file into the project and conduct the training and tests, after separating it into Training data and Test data. </br></br>

> 이 프로젝트를 시작할때의 주 목적은 사실 Pytorch 사용과 신경망 구현에 익숙해지기 위함이기도 했지만, 사실 Torch 라이브러리가 사용하는 여러 함수들이 어떤 데이터를 
  어떤 형식으로 받아와 어떻게 동작하는 지를 자세히 알아보고 싶은 것도 있었습니다. 그래서 생각한 것이 Torch가 자체적으로 지원하는 MNIST 데이터가 아닌, 컴퓨터 로컬
  디렉토리에 준비된 매트랩 형식의 MNIST 파일을 불러와 학습용 데이터 및 레이블, 그리고 테스트용 데이터 및 레이블로 분류 후, 학습 및 테스트를 실행하는 식으로 진행
  했습니다. 

</br>
</br>
Classification of MNIST handwritten numbers using Pytorch and local MNIST.mat file
