# -*- coding: utf-8 -*-


import tensorflow as tf
import random
import numpy as np
from argparse import ArgumentParser
import IO_Part as game
import time
from tqdm import tqdm
import pickle


def get_options():
    parser = ArgumentParser()
    parser.add_argument('--MAX_EPISODE', type=int, default=1000,
                        help='최대 에피소드를 몇번 반복할건지')

    parser.add_argument('--ACTION_DIM', type=int, default=8,
                        help='한번에 몇번 액션을 취할건지')

    parser.add_argument('--OBSERVATION_DIM', type=int, default=120198,
                        help='한번에 볼 수 있는 상태의 수')

    parser.add_argument('--GAMMA', type=float, default=0.95,
                        help='Q Learning에서의 감산 펙터')

    parser.add_argument('--INIT_EPS', type=float, default=0.9,
                        help='무작위 표본 추출 작업에 대한 초기 확률')

    parser.add_argument('--FINAL_EPS', type=float, default=1e-5,
                        help='무작위 표본 추출 작업에 대한 최종 확률')

    parser.add_argument('--EPS_DECAY', type=float, default=0.95,
                        help='엡실론 감쇠율')

    parser.add_argument('--EPS_ANNEAL_STEPS', type=int, default=2,
                        help='엡실론 감쇠를 위한 간격 단계')

    parser.add_argument('--LR', type=float, default=1e-4,
                        help='학습 비율')

    parser.add_argument('--MAX_EXPERIENCE', type=int, default=2048,
                        help='경험 재생 메모리의 크기')

    parser.add_argument('--BATCH_SIZE', type=int, default=128,
                        help='미니 배치의 크기')

    parser.add_argument('--H1_SIZE', type=int, default=512,
                        help='히든 레이어 1의 크기')

    parser.add_argument('--H2_SIZE', type=int, default=512,
                        help='히든 레이어 2의 크기')

    parser.add_argument('--H3_SIZE', type=int, default=256,
                        help='히든 레이어 3의 크기')

    parser.add_argument('--H4_SIZE', type=int, default=256,
                        help='히든 레이어 4의 크기')

    parser.add_argument('--H5_SIZE', type=int, default=256,
                        help='히든 레이어 5의 크기')

    parser.add_argument('--H6_SIZE', type=int, default=128,
                        help='히든 레이어 6의 크기')

    parser.add_argument('--H7_SIZE', type=int, default=128,
                        help='히든 레이어 7의 크기')

    parser.add_argument('--C1_SIZE', type=int, default=128,
                        help='Convolution Layer 1의 크기')

    parser.add_argument('--C2_SIZE', type=int, default=128,
                        help='Convolution Layer 2의 크기')

    parser.add_argument('--DROPOUT_RATE', type=float, default=0.7,
                        help='dropout 비율')

    parser.add_argument('--IMAGE_WIDTH', type=int, default=402,
                        help='가로')

    parser.add_argument('--IMAGE_HEIGHT', type=int, default=299,
                        help='세로')

    options = parser.parse_args()
    return options

'''
option은 각족 여러가지 선언이나 정의가 되어있는 곳입니다. 각 선언마다 help가 적혀있으므로 따로 설명하지는 않겠습니다. 그 중에서 DNN을 정의하는 부분에 대해서는 설명할 필요가 있을 것 같습니다. 작성자는 다음 홈페이지를 참고해서 neural network를 구성했다고 합니다.
http://www.nervanasys.com/demystifying-deep-reinforcement-learning/

hidden layer는 총 3개로서 2개이상이면 DNN이라고 부르기 때문에 DNN입니다. 각 각의 layer는 128개의 node들로 구성이 되어있습니다. activation function으로는 세 개의 hidden layer들은 ReLu function을 이용하고 있고 마지막에 output을 내는 함수에는 TensorFlow의 Squeeze 함수를 사용하고 있습니다. Input으로는 Observation
x : position of cart on the track
θ : angle of the pole with the vertical
dx/dt : cart velocity
dθ/dt : rate of change of the angle
이 들어가게 되고 output으로는 [1,0]아니면 [0,1]의 행렬이 나오게 됩니다. 이것은 action을 의미하는데 왼쪽으로 impact force을 주거나 오른쪽으로 주거나를 의미합니다.
'''


class QAgent:

    # A naive neural network with 3 hidden layers and relu as non-linear function.
    def __init__(self, options): # 초기화 부분

        self.W_conv1 = self.weight_variable([5, 5, 1, options.C1_SIZE])
        self.B_conv1 = self.bias_variable([options.C1_SIZE])

        self.W_conv2 = self.weight_variable([5, 5, options.C1_SIZE, options.C2_SIZE])
        self.B_conv2 = self.bias_variable([options.C2_SIZE])

        self.W1 = self.weight_variable([options.OBSERVATION_DIM, options.H1_SIZE])
        self.b1 = self.bias_variable([options.H1_SIZE])

        self.W2 = self.weight_variable([options.H1_SIZE, options.H2_SIZE])
        self.b2 = self.bias_variable([options.H2_SIZE])

        self.W3 = self.weight_variable([options.H2_SIZE, options.H3_SIZE])
        self.b3 = self.bias_variable([options.H3_SIZE])

        self.W4 = self.weight_variable([options.H3_SIZE, options.H4_SIZE])
        self.b4 = self.bias_variable([options.H4_SIZE])

        self.W5 = self.weight_variable([options.H4_SIZE, options.H5_SIZE])
        self.b5 = self.bias_variable([options.H5_SIZE])

        self.W6 = self.weight_variable([options.H5_SIZE, options.H6_SIZE])
        self.b6 = self.bias_variable([options.H6_SIZE])

        self.W7 = self.weight_variable([options.H6_SIZE, options.H7_SIZE])
        self.b7 = self.bias_variable([options.H7_SIZE])

        self.W8 = self.weight_variable([options.H7_SIZE, options.ACTION_DIM])
        self.b8 = self.bias_variable([options.ACTION_DIM])


    '''
    Q Agent class. 멘 처음 텐서플로우를 초기화 하는 부분
    3개의 hiddenlayer를 가진 Neural Network생성 W1,2,3,4는 weight, B1,2,3,4는 Bias를 나타냄
    Hidden Unit의 갯수는 Argument에 의해 정의. Default = 128
    '''

    def xavier_initializer(self, shape):
        dim_sum = np.sum(shape)
        if len(shape) == 1:
            dim_sum += 1
        bound = np.sqrt(6.0 / dim_sum)
        return tf.random_uniform(shape, minval=-bound, maxval=bound) # bound 절대값을 중심으로 Random 형태로 반환
    '''
    세개의 Hidden Layer 존재. 각 layer사이에는 Weight 행렬 존재 w1,2,3인데 이걸 초기화 해주는 단계
    초기화 방법으론 Xavier Initializition을 사용

    http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization
    Xavier 초기화가 중요한 이유는 무엇입니까?
    짧게 말하자면, 신호가 네트워크 깊숙이 도달하는 데 도움이됩니다.
    네트워크의 가중치가 너무 작으면 신호가 너무 작아서 유용하지 않을 때까지 각 레이어를 통과하면서 신호가 축소됩니다.
    네트워크의 가중치가 너무 크면 신호가 너무 커질 때까지 각 레이어를 통과하면서 신호가 커집니다.
    Xavier 초기화는 가중치가 '바로 올바르다'는 것을 확인하여 많은 레이어를 통해 합리적인 범위의 값으로 신호를 유지합니다.
    이보다 더 나아 가기 위해서는 소량의 통계가 필요합니다. 특히 무작위 배포 및 분산에 대해 알아야합니다.
    '''

    # Tool function to create weight variables
    def weight_variable(self, shape): # Weight 초기화
        return tf.Variable(self.xavier_initializer(shape)) # Weight을 xavier로 초기화 하여 리턴

    # Tool function to create bias variables
    def bias_variable(self, shape): # Bias 초기화
        return tf.Variable(self.xavier_initializer(shape)) # Bias를 xavier로 초기화 하여 리턴
    '''
    실제 Bias와 Weight를 xavier로 초기화 시켜주는 함수
    '''

    def conv2d(self,x,W):
        return tf.nn.conv2d(x,W,strides=[1,1,1,1], padding='SAME')

    def max_pool_2x2(self,x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Add options to graph
    def add_value_net(self, x, options):
        '''
        x_image = tf.reshape(x,[-1,28,28,1])
        h_conv1 = tf.nn.relu(self.conv2d(x_image, self.W_conv1)+ self.B_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)

        h_conv2 = tf.nn.relu(self.conv2d(h_pool1,self.W_conv2) + self.B_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)
        '''
        observation = tf.placeholder(tf.float32, [None, options.OBSERVATION_DIM]) # Place holder로 observation으로 값을 넣음.
        _h1 = tf.nn.relu(tf.matmul(observation, self.W1) + self.b1) # ReLu는 활성화 함수
        h1 = tf.nn.dropout(_h1, options.DROPOUT_RATE)
        _h2 = tf.nn.relu(tf.matmul(h1, self.W2) + self.b2)
        h2 = tf.nn.dropout(_h2, options.DROPOUT_RATE)
        _h3 = tf.nn.relu(tf.matmul(h2, self.W3) + self.b3)
        h3 = tf.nn.dropout(_h3, options.DROPOUT_RATE)
        _h4 = tf.nn.relu(tf.matmul(h3, self.W4) + self.b4)
        h4 = tf.nn.dropout(_h4, options.DROPOUT_RATE)
        _h5 = tf.nn.relu(tf.matmul(h4, self.W5) + self.b5)
        h5 = tf.nn.dropout(_h5, options.DROPOUT_RATE)
        _h6 = tf.nn.relu(tf.matmul(h5, self.W6) + self.b6)
        h6 = tf.nn.dropout(_h6, options.DROPOUT_RATE)
        _h7 = tf.nn.relu(tf.matmul(h6, self.W7) + self.b7)
        h7 = tf.nn.dropout(_h7, options.DROPOUT_RATE)
        Q = tf.squeeze(tf.matmul(h7, self.W8) + self.b8)
        return observation, Q
    '''
    http://pythonkim.tistory.com/40
    Backward 연산에서 결과를 전달할 때 Sigmoid를 사용 근대 sigmoid는 값이 0에서 1사이에서 심하게 변형을 함
    이렇게 작아진 값을 가지고 변형을 하기가 힘듬 (깊은 네트워크에서)

    이런 문제를 Vanishing Gradient라 부름
    Layer를 지날때마다 최초 값 보다 현저하게 작아지기 때문에 값을 전달해도 의미를 가질 수가 없다

    hinton 교수가 Sigmoid 대신 Relu 함수를 제안
    Relu함수는 0보다 작을 때는 0을 사용하고 0보다 큰 값에서는 해당 값을 사용하는 방법.
    음수에 대해 값이 바뀌지만 양수에 대해서는 그대로
    ReLu를 함수로 구현함녀 max(0,x)-> 0과 현재값중에 큰 값을 선택하면 됨.

    Tensorflow에서 ReLU를 적용하는 방법은 Sigmoid 대신에 Relu 함수로 대신함
    전달 값을 처리하는 방식이 달라졌기 때문에 전혀 수정하지 않아도 됨
    ReLu함수는 Rectified Linear Unit의 약자로 기존의 Linear함수인 sigmoid를 대체했다는 의미를 가짐

    Activation함수는 여러가지가 있음.
    Tanh -> Sigmoid 재활용 범위를 -1에서 1로 넓힘
    ReLu -> 음수에 대해서만 0으로 처리하는 함수
    Leky ReLu -> 음수에 대해 1/10으로 값을 줄여서 사용하는 함수
    ELU-> ReLu를 0이 아닌 다른 값을 기준으로 사용하는 함수
    maxout -> 두개의 W와 b중에서 큰 값이 나온 것을 사용하는 함수


    Network의 세부사항 정의.
    마지막에는 Linear Function(squeeze)가 들어가고 나머지는 모두 ReLu를 사용.
    Squeez는 차원 크기가 1인 부분을 제거 (없어도 되는 듯)
    (왜 들어간지 모르겠음)

    '''
    # Sample action with random rate eps
    # 액션을 뽑아주는 함수
    def sample_action(self, Q, feed, eps, options): # 위에서 정의한 Q, feed, Epsilon, options를 받아와서 action을 정의해줌

        act_values = Q.eval(feed_dict=feed) # Feed를 먹이고 Q를 실행하여 나온 값(Q value)을 act_value에 삽입
        if random.random() <= eps: # 랜덤 값을 생성해서 epsilon 보다 작으면 random action을 수행
            # action_index = env.action_space.sample() #openAI Gym에서 샘플 액션을 받아올 때 사용
            action_index = random.randrange(options.ACTION_DIM) #ACTION_DIM 크기만큼에서 랜덤 한 수를 추출하여 action_index로
        else: # epsilon 보다 크다면
            action_index = np.argmax(act_values) #act_values에 저장된 Q Value 중에서 좀 더 큰 Q value를 갖는 action을 선택
            # Argmax는 f(x)에서 f(x)를 최대값을 가지게하는 x값을 찾는 함수
        action = np.zeros(options.ACTION_DIM) # Action이란 리스트를 생성후 ACTION_DIM 만큼 0으로 채운다
        action[action_index] = 1 # Action_index 넘버의 Action 리스트를 1로 변경

        return action #Action 리스트를 반환한다

    '''
    액션을 뽑아주는 함수 위에서 정의한 Q, feed, Epsilon, options을 받아와서 action을 정의해줌
    Epsilon greedy한 방법을 이용하고 default로 정해진 옵션으로 처음 Epsilon은 1.
    (parser.add_argument('--INIT_EPS', type=float, default=1.0,)

    http://sanghyukchun.github.io/96/
    ε-greedy라는 알고리즘이 있다.
    이 알고리즘은 1−ε의 확률로 지금까지 관측한 arm 중에 가장 좋은 arm을 고르고 (exploitation),
    ε의 확률로 나머지 arm 중에서 random한 arm을 골라서 play하는 (explore) 알고리즘이다.

    http://www.modulabs.co.kr/RL_library/2621
    (1) Policy Iteration처럼 evaluation을 끝까지하지 않고 한 번의 episode로 인해 얻은 정보를 토대로 현재의 policy를 평가해줍니다.

    (2) evaluation하고 대상이 state-value function이 아니고 action-value function입니다.
    model free control이 되려면 MDP의 model을 몰라도 할 수 있어야하는데 state-value function을 사용하면
    greedy improvement할 때 model이 필요하기 때문입니다.

    (2) Greedy policy improvement가 아니고 epsilon이 그 앞에 들어가는데 이유는 exploration문제에 있습니다.(최적)
    진정한 답을 얻기위해서는 충분하게 경험해보는 것이 필요한데 그냥 greedy improvement할 경우에는
    얻은 답이 optimal한 해답이라고 장담할 수 없기 때문에 일정한 모험의 요소를 추가한 것

    두 가지 on-policy control(MC and TD)에서 epsilon greedy하게 improve를 하는데
    optimal한 값으로 가기 위해서는 epsilon이 0으로 수렴하는 구조여야합니다.
    그렇게 되면 충분한 exploration(최적)을 보장할 수 없으므로 여기에서 off-policy learning이 등장합니다.

    off-policy learning이란 두 가지  policy를 동시에 사용할 경우를 말합니다.
    즉, learning에 사용되는 policy는 greedy하기 improve를 하고 움직일 때는 현재의 Q function을 토대로 epsilon greedy하게 움직입니다.

    다음 가능한 Q function중에서 최대를 뽑아서 현재 Q function을 update하고
    움직일 때는 max값을 알면서도 가끔 다른 행동을 하게 됩니다.

    Q.eval
    https://www.tensorflow.org/versions/r0.12/resources/glossary
    eval()
    값을 결정하는데 필요한 그래프 계산을 트리거링하며 Tensor의 값을 반환하는 Tensor의 메서드입니다.
    세션에서 시작된 그래프에서 Tensor의 eval()을 호출하기만 하면됩니다.

    '''

class progress():
    def open(self):
        f = open('./Progress_result.txt','rb')
        episode = pickle.load(f)
        f.close()
        return episode

    def save(survive_time):
        survive_result = []
        survive_result= progress.open(0)
        #survive_result = survive_time
        survive_result.append(survive_time)
        #To save
        f = open('./Progress_result.txt','wb')
        pickle.dump(survive_result,f)
        f.close()
        print("==이태까지의 진행 과정을 저장했습니다!==")
#save class end

# 실제적인 학습 부분

def train(env):
    # Define placeholders to catch inputs and add options
    options = get_options() # 위에서 정의한 get_options()를 통해 옵션을 정의함
    agent = QAgent(options) # QAgent class를 생성하여 agent라는 이름으로 넣음
    sess = tf.InteractiveSession() # Interactive Session을 생성
    obs, Q1 = agent.add_value_net(0,options) # Observation을 넣으면 Q value가 나오는 함수 add_value_net 형태만 지정해줌
    act = tf.placeholder(tf.float32, [None, options.ACTION_DIM]) #float32 타입으로 ACTION_DIM 크기로 action 크기 결정
    rwd = tf.placeholder(tf.float32, [None, ]) # reward 크기 결정
    next_obs, Q2 = agent.add_value_net(0,options) # Observation을 넣으면 Q value가 나오는 함수 add_value_net 형태만 지정해줌. Q1과 같은 형태
    values1 = tf.reduce_sum(tf.multiply(Q1, act), reduction_indices=1)
    values2 = rwd + options.GAMMA * tf.reduce_max(Q2, reduction_indices=1)
    # reduction_indices=1 는 행끼리 처리하라는 reduce 연산의 옵션
    loss = tf.reduce_mean(tf.square(values1 - values2))
    '''
    [Deep Q Learning with Experience Replay]
    Neural Network를 update할 때에는 특정 기준으로 update를 하는데 그 기준이 loss fuction.
    Q(s',a')에 max를 취함으로서 greedy한 policy로 업데이트하게 되고 그렇게 Q learning이 됨
    한 번 update하는데  training set 256를 사용. 256개의 value 1과 value 2를 계산하는데
    value 2안에 max가 들어가있음. 256개의 value 1과 value 2의 차이의 제곱의 평균이 loss function으로 정의됩니다.
    '''
    train_step = tf.train.AdamOptimizer(options.LR).minimize(loss)
    '''
    Neural Networks의 parameter를 update하는 방법은 back Propagation. 그 정도를 정하는게 optimizer.
    DQN에서는 Stochastic Gradient descent를 사용하지만  실제로는 Adma과 RMSprop를 사용.
    gradient를 가진다는 것은 어떠한 속도를 가진다고도 생각할 수 있는데
    물리라고 생각해보면 운동하는 물체의 운동 관성, 즉 momentum을 고려해준다는 것입니다.
    그러한 방법을 momentum update라고 합니다.
    또한 gradient를 따라서 descent하는 정도를 learning rate라고 하는데 그 learning rate가 정해져 있는 것이 아니고
    시간이 지날수록 decay시키는 방법을 Adaprop이라고 합니다.
    이때 그 decayed 된 learning rate를 일종의 필터링을 거쳐서 사용하는 것이 RMSprop이고
    거기에 momentum효과를 더하면 그것이 Adam optimizer가 됩니다.
    Loss를 최소화 하는 방향으로 Learning rate(1e-4)를 가지고 Adam 최적화 알고리즘을 사용하여 학습한 결과를 train.step에 저장
    '''
    sess.run(tf.global_variables_initializer()) # 모든 변수를 초기화 해주는 부분

    #network 저장 로드 부분
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state("checkpoints")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")
    '''
    모델을 저장하고 복구하기 위한 가장 쉬운 방법은 tf.train.Saver 오브젝트를 이용하는 것.
    이 오브젝트를 통해 그래프 전체 변수 또는 그래프의 지정된 리스트의 변수에 save와 restore 오퍼레이션(op)을 추가할 수 있음
    Saver 오브젝트는 이 오퍼레이션을 실행하는 메서드와 읽고 쓰는 체크포인트 파일의 지정된 경로를 제공합니다.
    saver.save(sess, "/tmp/model.ckpt")
    saver.restore(sess, "/tmp/model.ckpt")
    같은 식으로 사용
    '''
    #로컬 변수에 대한 초기화 및 정의
    feed = {}
    eps = options.INIT_EPS
    global_step = 0 # global step 정의 및 초기화
    exp_pointer = 0 # queue의 순서를 나타내는 포인터
    '''
    experience reply를 할때 같은 episode안에서 순서대로 학습하면 편향되는 효과를 가짐
    MAX_EXPRIENCE를 정의하고 거기에 training set을 하나씩 저장해서 랜덤으로 추출하여 학습
    training set은 obs, act, rwd, next_obs queue로 이루어져있음.
    '''
    # The replay memory
    obs_queue = np.empty([options.MAX_EXPERIENCE, options.OBSERVATION_DIM])  #행이 MAX_EXPERIENCE이고 열이 OBSERVATION_DIM인 빈 행렬을 생성
    act_queue = np.empty([options.MAX_EXPERIENCE, options.ACTION_DIM])
    rwd_queue = np.empty([options.MAX_EXPERIENCE])
    next_obs_queue = np.empty([options.MAX_EXPERIENCE, options.OBSERVATION_DIM])
    CNN_queue = np.empty([options.MAX_EXPERIENCE,1])

    temp_obs_queue = np.empty([options.MAX_EXPERIENCE, options.OBSERVATION_DIM])   #학습 되는 내용을 저장하기 위해
    temp_act_queue = np.empty([options.MAX_EXPERIENCE, options.ACTION_DIM])
    temp_rwd_queue = np.empty([options.MAX_EXPERIENCE])
    temp_next_obs_queue = np.empty([options.MAX_EXPERIENCE, options.OBSERVATION_DIM])


    living_time_queue = []
    #i_episode = 0
    i_episode = int(len(progress.open(0))) +1
    print("==이태까지의 진행 상황을 읽어옵니다!==")
    print("=={}==".format(i_episode))

    # 에피소드 루프 정의
    while 1:
        obs1, obs2 = game.reset_env()
        #observation = np.append(obs1,obs2)
        observation = obs1

        obs_to_1dim = np.reshape(observation, (-1,)) #차원을 낮춘다
        done = False # episode가 끝나기 전까진 done = false
        sum_loss_value = 0
        temp_pointer = 0
        living_time = 0
        axis=[0,0]
        boundray_contact = False

        #Exp Loop
        while not done: #에피소드가 끝나지 않았다면
            global_step += 1 #global step을 1 더해주고
            #if i_episode % options.EPS_ANNEAL_STEPS == 0 and eps > options.FINAL_EPS:
            if i_episode % 2 == 0 and eps > options.FINAL_EPS:
                #에피소드가 EPS_ANNEAL_STEPS를 지날때마다 그리고 엡실론이 최종 엡실론 보다 크면
                eps = eps * options.EPS_DECAY # 엡실론에 감쇠 비율을 적용한다
            #if end
            #print("size is : {} x {}".format(np.size(obs1,1),np.size(obs1,0)))
            temp_obs_queue[temp_pointer] = obs_to_1dim # Observation 값을 obs_queue에 저장 exp_pointer는 큐의 순서를 지칭함
            '''
            action은 agent 클래스 QAgent(options) 에서 sample_action 함수를 통해서 결정
            Epsilon greedy하게 현재 observation이 DQN을 통과하여 나온 Q값을 바탕으로 action을 결정함)

            def sample_action(self, Q, feed, eps, options)
            Q는 Q1. Q1은 agent.add_value_net(options) 에서 나온 결과값.

            add_value_net는 Q Agent 초기화 부분에서 W1,B1... 등 클래스 그 자신을 가져와서 수행
            Feed는 obs에
            '''
            converted_observation = np.reshape(observation, (1,-1))
            action = agent.sample_action(Q1, {obs: converted_observation}, eps, options)  # Action 추출
            action = np.argmax(action)

            obs1, obs2, reward, done, living_time, axis = game.Game_env(action,axis)
            #OpenAI Gym. _는 info 부분. argmax는 최대 구성요소의 인덱스를 반환합니다

            #viewing observation
            if living_time == 0:
                boundray_contact = True

            #cv2.imshow('obs1', obs1)
            #if cv2.waitKey(25) & 0xFF == ord('q'):
            #    cv2.destroyAllWindows()

            #cv2 end

            #observation = np.append(obs1,obs2)
            observation = obs1
            obs_to_1dim = np.reshape(observation, (-1,))

            if done: #done이 뜬 값
                reward = -1
            #if done end
            temp_act_queue[temp_pointer] = action
            temp_next_obs_queue[temp_pointer] = obs_to_1dim  # Observation을 저장하고
            temp_rwd_queue[temp_pointer] = reward  # reward queue에 reward 값을 넣고
            temp_pointer += 1  # queue의 index pointer를 1 증가시킨다

        #step loop 종료


        #스코어 누적 과정
        for i in range(temp_pointer):
                if exp_pointer != options.MAX_EXPERIENCE:
                    if living_time >=0.5:
                        obs_queue[exp_pointer] = temp_obs_queue[i]

                        act_queue[exp_pointer] =  temp_act_queue[i]
                        next_obs_queue[exp_pointer] =  temp_next_obs_queue[i]
                        rwd_queue[exp_pointer] = temp_rwd_queue[i]
                        exp_pointer += 1
                #exp_point if end
                else:
                    break
                        # for end

                # Print Result
        print("====== Survive time = {}, Exp size = {} ======".format(living_time, exp_pointer))
        living_time_queue.append(living_time)
        '''
        # Learn Part
        if exp_pointer == options.MAX_EXPERIENCE:
            print("====== {} ======".format(time.ctime()))
            print("학습을 시작합니다.")
            for _ in tqdm(range(temp_pointer)):
                rand_indexs = np.random.choice(options.MAX_EXPERIENCE, options.BATCH_SIZE)
                # max_exprience에서 batch size만큼 랜덤으로 추출하여 rand indexs에 넣는다
                feed.update({obs: obs_queue[rand_indexs]})
                feed.update({act: act_queue[rand_indexs]})
                feed.update({rwd: rwd_queue[rand_indexs]})
                feed.update({next_obs: next_obs_queue[rand_indexs]})
                # 랜덤한 indexs에 따라 각각 obs,act,rwd,next obs를 update
                step_loss_value, _ = sess.run([loss, train_step], feed_dict=feed)
                # Use sum to calculate average loss of this episode
                sum_loss_value += step_loss_value
                # feed를 먹이고 loss함수를 가지고 train_step을 진행. 그 값을 step_loss_value로 저장한다
                np.delete(obs_queue, [rand_indexs])
                np.delete(act_queue, [rand_indexs])
                np.delete(rwd_queue, [rand_indexs])
                np.delete(next_obs_queue, [rand_indexs])
            # for end
            exp_pointer = 0  # 학습 포인터 초기화
            print("====== {} ======".format(time.ctime()))
            print("학습 완료!")

            #저장부분
            saver.save(sess, './checkpoints/' + 'dqn', global_step=global_step)  # ./checkpoints/ 폴더에 저장
            #print("== 신경망을저장했습니다!!==")  # save!라고 말함
            i_episode += 1
            living_time = 0

            for i in range(len(living_time_queue)):
                living_time += living_time_queue[i]

            living_time = living_time / len(living_time_queue)
            print("====== {} ======".format(time.ctime()))
            print("====== Episode {} end, average survive time is {}======".format(i_episode, living_time))
            progress.save(living_time)
            living_time_queue = []
            if i_episode % 20 == 0 and eps < options.FINAL_EPS:
                eps = options.INIT_EPS # 저장하면 엡실론 초기화
            #저장부분 끝

        #Learn Part end

    #episode end

#train end

if __name__ == "__main__":
    train(game)