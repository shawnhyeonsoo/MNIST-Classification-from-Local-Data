# MNIST-Classification-from-Local-Data </br>
_____________________________________________________________
This project is an implementation of classification of MNIST handwritten numbers data taken from a local matlab file,
with the help of Pytorch. </br> *NOTE: THE PROJECT WAS COMPLETED AS AN INDIVIDUAL WHO STILL HAS A LONG WAY TO GO; THE 
CODE MAY NOT BE OF THE BEST QUALITY, SO PLEASE DO NOT HESITATE TO DROP ADVICE!*

The Convolutional Neural Network (CNN) is composed of the following: </br>

- an input layer (784 nodes)
- first convolution layer
- ReLU
- first max-pooling layer
- second convolution layer
- ReLU
- second max-pooling layer
- third fully-connected layer
- output layer (10 nodes)


</br>

> 본 프로젝트는 Pytorch를 이용하지만 torch에서 사용 및 다운로드를 지원하는 MNIST 파일이 아닌 컴퓨터 로컬 디렉토리에 저장되어 있는 MNIST.mat 형식의 파일을
  불러와 텐서 형식으로 불러온 후, 학습 및 테스트하는 프로그램입니다. 
*본 프로그램은 개인이 공부하면서 작성한 코드를 포함하므로, 최적의 상태를 가진 코드는 아닐 수 있습니다. 고칠 점이 보이신다면 언제든지 조언을 남겨주세요! 감사합니다.*
</br>
</br>
이 합성곱 신경망으로 구현된 코드는 다음과 같이 구성됩니다:</br>

- 입력층
- 1번 합성곱층
- 1번 ReLU
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
The code for the Convolutional Neural Network is pretty straightforward; the Conv2d function is used for the Convolutional
layer, and the MaxPool2d function for the Pooling layer. The difficulty came when trying to load the matlab file in a way
such that the program could conduct training and test on. Finding out that scipy.io library had to be imported in order to
load a matlab file took almost no time at all; the tragedy was trying to turn the data loaded from the matlab file into tensor
data sets. </br>
</br>
As mentioned, the matlab file was loaded through 'loadmat' after getting the relative directory of the local MNIST.mat file
from os.getcwd(). The train data and label were extracted through means such as:

```python
trainx = mnist_raw['trainX']
```

at first. After means of trial and error, the torch.tensor() functions and torch.from_numpy().long().to(device) functions were
used to transform the data into tensors. Followed by reshaping the trainx, the train data, into appropriate dimensions. 
Finding the right dimensions really took me a long time; it had been a repetition of 'CompilationError' on the screen after
uncountable trials. Still, it was successfully managed, and what followed was smooth; turning the train data and labels into
tensor data set followed by transformation into 'train_loader', the data_loader to be used in the process of training. 
Although it was a troublesome and time-taking process, I believe that it had been a meaningful experience as I had the 
chance to examine the relevant libraries and variables thoroughly. 
The Network showed 98.69999885559082% of accuracy.
</br>
</br>
</br>

> 프로젝트 코드에 있는 합성층 레이어 및 학습, 테스트 함수들은 복잡할 것이 없다. 인터넷을 비롯한 여러 곳에서 쉽게 찾아 볼 수 있으며, 가장 기본적인 코드를 참고하며
  진행했다. 가장 난이도가 있었던 부분이 매트랩 파일에서 각종 데이터를 프로젝트의 여러 함수가 올바르게 인식할 수 있도록 변환해주는 과정이었다. 매트랩 파일을 파이썬
  으로 불러오기 위해 scipy.io 라이브러리가 필요하다는 것은 금방 찾을 수 있었다. 그러나, 텐서로 변환하며 dimension이 맞도록 직접 설정해주어야 했는데, 이 과정
  에서 시간을 많이 소비했다. 로컬 디렉토리에 같은 폴더에 준비되어 있는 MNIST.mat 파일을 불러오기 위해서 os.getcwd() 함수를 사용하여 현재 디렉토리를 구한 후,
  상대적 거리를 변수로 지정해 loadmat 함수를 이용하여 쉽게 불러올 수 있었다. </br>
  
  
> 학습용 데이터와 레이블을 나누는 과정에서는 위에 나온 *trainx = mnist_raw['trainX']* 와 같은 함수를 이용하여 쉽게 추출할 수 있었으나, 앞서 말했듯이
  올바른 배열의 차원을 지정해주는 것은 인터넷을 찾아도 쉽게 나오지 않았으며, pytorch 에서 자체적으로 지원하는 MNIST 파일과 비교하며 공부했다. 결국 시행착오
  끝에 올바른 형식을 구했고, 그 후엔 데이터 로더 등의 함수로 코드에 맞게 변환하는 것은 어렵지 않았다. </br>
 

> 결국 프로젝트를 제대로 끝맺을 수 있었는데, 비록 큰 스케일과 복잡도를 가지진 않은 프로젝트에 비해 시간과 노력이 많이 들어간 것 같기는 하지만, 연관되는 라이브러리
  와 변수들을 제대로 관찰하고 공부할 수 있어 과정에서 배운 것이 많은 것 같다. 앞으로 어떤 프로젝트를 하게 될 진 모르겠지만 이와 같이 끈기를 갖고 하다보면 되지 
  않을까라는 자신감도 생긴 것 같다. 
  결과적으로 이 네트워크는 98.69999885559082% 의 정확도를 보였다. 
  
 
 </br>
 </br>
 </br>
 </br>
 </br>
 </br>
 </br>
 
 본 프로젝트에 있어서 <https://wikidocs.net/63618>의 글을 참고했습니다
 
 </br>
___________________________________________________________________________________________
Classification of MNIST handwritten numbers using Pytorch and local MNIST.mat file
