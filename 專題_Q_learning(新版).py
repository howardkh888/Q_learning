import math
import cmath
import numpy as np
import matplotlib.pyplot as plt

def Lambda(a,b,d):
    PL_dB = a + 10*b*math.log10(d) + 5.8*np.random.randn()
    return math.sqrt(10**((PL_dB-30)/10))
def HLos(Nr,Nt):
    return math.sqrt(1)*math.sqrt(1/2)*(np.random.randn(Nr,Nt) + 1j*np.random.randn(Nr,Nt))
def HNLos(Nr,Nt):
    L = 2
    HNLos = 0
    for i in range(0,L):
        HNLos = HNLos + math.sqrt(0.5)*math.sqrt(1/2)*(np.random.randn(Nr,Nt) + 1j*np.random.randn(Nr,Nt))
    return HNLos
def DataRate(cr,ct,p1,p2,Noise):
    Rr = math.log2(1 + abs(cr)**2*p1/(abs(cr)**2*p2+Noise))
    Rt = math.log2(1 + abs(ct)**2*p2/(abs(ct)**2*p1+Noise))
    return Rr + Rt
def DataRate_r(cr,p1,p2,Noise):
    Rr = math.log2(1 + abs(cr)**2*p1/(abs(cr)**2*p2+Noise))
    return Rr
def DataRate_t(ct,p1,p2,Noise):
    Rt = math.log2(1 + abs(ct)**2*p2/(abs(ct)**2*p1+Noise))
    return Rt

def Reward(R_old,R_new,Rt_new):
    if (R_new > R_old) and (Rt_new >= 1) and ((R_new - Rt_new) >= 1):
        rt = 2
    elif (R_new > R_old):
        rt = 1
    elif (R_new < R_old) and (Rt_new >= 1) and ((R_new - Rt_new) >= 1):
        rt = -1
    elif (R_new < R_old):
        rt = -2
    else:
        rt = 0
    return rt

def transmission_coefficient(M,state_phi,phi_r_a,state_btea,btea_r_a):
    phi_t = np.zeros(M)
    btea_t = np.zeros(M)
    diag_element = np.zeros(M,dtype=complex)
    for i in range(M):
        #調整phi_r後調phi_t
        if(np.random.rand() < 0.5):
             phi_t[i] = phi_r_a[int(state_phi[i])] + 0.5*math.pi
        else:
             phi_t[i] = phi_r_a[int(state_phi[i])] - 0.5*math.pi
        #調整phi_t在0到2pi之間(修改elif)
        if(phi_t[i] >= 2*math.pi):
            phi_t[i]  = phi_r_a[int(state_phi[i])] - 0.5*math.pi
        elif(phi_t[i] < 0): #Question
            phi_t[i] = phi_r_a[int(state_phi[i])] + 0.5*math.pi
        btea_t[i] = 1 - btea_r_a[int(state_btea[i])]
        diag_element[i] = complex(math.sqrt(btea_t[i]))*cmath.exp(1j*phi_t[i])
    return np.diag(diag_element)    
def reflection_coefficient(M,state_phi,phi_r_a,state_btea,btea_r_a):
    phi_r = np.zeros(M)
    btea_r = np.zeros(M)
    diag_element = np.zeros(M,dtype=complex)
    for i in range(M):
        phi_r[i] = phi_r_a[int(state_phi[i])]
        btea_r[i] = btea_r_a[int(state_btea[i])]
        diag_element[i] = complex(math.sqrt(btea_r[i]))*cmath.exp(1j*phi_r[i])
    return np.diag(diag_element)  

def moving_average(x, w):
    return np.convolve(x, np.ones(w)/w, 'valid')
    
#參數設定
M=16
Noise = 10**((-80-30)/10) # sigma^2 = -80dBm(轉成W,power)
Pmax = 1 #(1W = 30dBm, maximum power)


#Rician fading channel
BS = np.array([0,0,10])
STAR_RIS = np.array([50,10,10])
UEr = np.array([30,10,1])
UEt = np.array([90,15,1])

a = 61.4
b = 2
e = 2

dis_d = 0
dis_g = 0
dis_vr = 0
dis_vt = 0
for i in range(3):
    dis_d = dis_d + (BS[i]-UEr[i])**2
    dis_g = dis_g + (BS[i]-STAR_RIS[i])**2
    dis_vr = dis_vr + (UEr[i]-STAR_RIS[i])**2
    dis_vt = dis_vt + (UEt[i]-STAR_RIS[i])**2
dis_d = math.sqrt(dis_d)
dis_g = math.sqrt(dis_g)
dis_vr = math.sqrt(dis_vr)
dis_vt = math.sqrt(dis_vt)

H_d = math.sqrt(1/Lambda(a,b,dis_d)) * (math.sqrt(e/(1+e))*HLos(1,1)+math.sqrt(1/(1+e))*HNLos(1,1))
H_g = math.sqrt(1/Lambda(a,b,dis_g)) * (math.sqrt(e/(1+e))*HLos(M,1)+math.sqrt(1/(1+e))*HNLos(M,1))
H_vr = math.sqrt(1/Lambda(a,b,dis_vr)) * (math.sqrt(e/(1+e))*HLos(1,M)+math.sqrt(1/(1+e))*HNLos(1,M))
H_vt = math.sqrt(1/Lambda(a,b,dis_vt)) * (math.sqrt(e/(1+e))*HLos(1,M)+math.sqrt(1/(1+e))*HNLos(1,M))

#Agent
#p1,p2,phi_r,btea_r
#State
p1_a = np.arange(0,Pmax+0.1,0.1)
p2_a = np.arange(0,Pmax+0.1,0.1)
phi_r_a = np.arange(0,2*math.pi+2*math.pi*0.1,2*math.pi*0.1)
btea_r_a = np.arange(0,1+0.1,0.1)

#Action
#p1_act = {+0.1,-0.1,0}
#p2_act = {+0.1,-0.1,0}
#phi_r_act = {2*math.pi*0.1,-2*math.pi*0.1,0}
#btea_r_act = {+0.1,-0.1,0}

#Reward
#Reward()

#Multi-agent Q-learning for STAR
#Set learning rate: alpha, discount factor: r, exploration rate: exp
alpha = 0.1 #不能太大，否則會掉到local maximum，越小更新則是越慢 要迭代更多次
r = 0.9  #是考慮未來獎勵對於現在影響的因子，是一個(0,1)之間的值。一般我們取0.9，能夠充分地對外來獎勵進行考慮。
exp = 0.3 #一開始以一定的機率隨意選擇action，隨著不斷學習，會以衰減函數來降低這個機率，按照Qtable來學習
#Initialize Q_table(st,at)
p1 = np.zeros((1,11,3))
p2 = np.zeros((1,11,3))
phi_r = np.zeros((M,11,3))
btea_r = np.zeros((M,11,3))

#Set iteration, Tmax
t=0
Tmax = 500
#初始化state
state_p1 = 1
state_p2 = 1
state_phi = np.full(M,5)
next_state_phi = np.full(M,0)
state_btea = np.full(M,5)
next_state_btea = np.full(M,0)
#儲存每次的Data Rate
D_Rate = np.zeros(Tmax)
D_Rate_Rt = np.zeros(Tmax)
D_Rate_Rr =  np.zeros(Tmax)
while(t<Tmax): 
    for i in range(4):
        if(i==0): #調整p1
            #Select action
            if(np.random.rand() < exp*math.exp(-1*t/100)):
                action = np.random.randint(3)
            else:
                action = np.argmax(p1[0,state_p1,:])
            #Perform action and acquire reward
            #a.找出next state(Perform action)
            if(action==0):
                next_state = state_p1 + 1
            elif(action==1):
                next_state = state_p1 - 1
            elif(action==2):
                next_state = state_p1
            if(next_state>10):
                next_state = next_state - 1
            elif(next_state<1):
                next_state = next_state + 1
            #調完p1發現不符合條件，重新調整p1的state * 
            if (p1_a[next_state] + p2_a[state_p2] > Pmax):
                next_p1 = round(p1_a[next_state] / (p1_a[next_state] + p2_a[state_p2]) * Pmax,1)
                next_p2 = round(p2_a[state_p2] / (p1_a[next_state] + p2_a[state_p2]) * Pmax,1)
                for j in range(11):
                    if next_p1 == p1_a[j]:
                        next_state = j
                    if next_p2 == p2_a[j]:
                        state_p2 = j
            #b.acquire reward
            #寫一個計算reflection_coefficient,transmission_coefficient的function
            tra = transmission_coefficient(M,state_phi,phi_r_a,state_btea,btea_r_a)
            ref = reflection_coefficient(M,state_phi,phi_r_a,state_btea,btea_r_a)
            cr = H_d + np.matmul(np.matmul(H_vr,ref),H_g)
            ct = np.matmul(np.matmul(H_vt,tra),H_g)
            R_old = DataRate(cr,ct,p1_a[state_p1],p2_a[state_p2],Noise)
            R_new = DataRate(cr,ct,p1_a[next_state],p2_a[state_p2],Noise)
            Rt_new = DataRate_t(ct,p1_a[next_state],p2_a[state_p2],Noise)#修改p1_a
            rt = Reward(R_old,R_new,Rt_new)
            #print(rt)
            #print(state_p1)
            #print(state_p2)
            #Observe next state s_t+1
            max_action = np.argmax(p1[0,next_state,:])
            #Update Q table
            p1[0][state_p1][action] = (1 - alpha) *  p1[0][state_p1][action] + alpha * (rt + r * p1[0][next_state][max_action])
            #切換到下個state
            state_p1 = next_state
        elif(i==1): #調整p2
            #Select action
            if(np.random.rand() < exp*math.exp(-1*t/100)):
                action = np.random.randint(3)
            else:
                action = np.argmax(p2[0,state_p2,:])#*
            #Perform action and acquire reward
            #a.找出next state(Perform action)
            if(action==0):
                next_state = state_p2 + 1 #*
            elif(action==1):
                next_state = state_p2 - 1 #*
            elif(action==2):
                next_state = state_p2 #*
            if(next_state>10):
                next_state = next_state - 1
            elif(next_state<1):
                next_state = next_state + 1
            #調完p2發現不符合條件，重新調整p2的state #* 
            if (p1_a[state_p1] + p2_a[next_state] > Pmax):
                next_p1 = round(p1_a[state_p1] / (p1_a[state_p1] + p2_a[next_state]) * Pmax,1)
                next_p2 = round(p2_a[next_state] / (p1_a[state_p1] + p2_a[next_state]) * Pmax,1)
                for j in range(11):
                    if next_p1 == p1_a[j]:
                        state_p1 = j
                    if next_p2 == p2_a[j]:
                        next_state = j
            #b.acquire reward
            #寫一個計算reflection_coefficient,transmission_coefficient的function
            tra = transmission_coefficient(M,state_phi,phi_r_a,state_btea,btea_r_a)
            ref = reflection_coefficient(M,state_phi,phi_r_a,state_btea,btea_r_a)
            cr = H_d + np.matmul(np.matmul(H_vr,ref),H_g)
            ct = np.matmul(np.matmul(H_vt,tra),H_g)
            R_old = DataRate(cr,ct,p1_a[state_p1],p2_a[state_p2],Noise)
            R_new = DataRate(cr,ct,p1_a[state_p1],p2_a[next_state],Noise)
            Rt_new = DataRate_t(ct,p1_a[state_p1],p2_a[next_state],Noise)#修改p2_a
            rt = Reward(R_old,R_new,Rt_new)
            #print(rt)
            #Observe next state s_t+1 #*
            max_action = np.argmax(p2[0,next_state,:])
            #Update Q table #*
            p2[0][state_p2][action] = (1 - alpha) *  p2[0][state_p2][action] + alpha * (rt + r * p2[0][next_state][max_action])
            #切換到下個state
            state_p2 = next_state
        elif(i==2): #調整phi
            for k in range(M):
                #Select action
                if(np.random.rand() < exp*math.exp(-1*t/100)):
                    action = np.random.randint(3)
                else:
                    action = np.argmax(phi_r[k,state_phi[k],:])
                #Perform action and acquire reward
                #a.找出next state(Perform action)
                if(action==0):
                    next_state_phi[k] = state_phi[k] + 1
                elif(action==1):
                    next_state_phi[k] = state_phi[k] - 1
                elif(action==2):
                    next_state_phi[k] = state_phi[k]
                if(next_state_phi[k]>10):
                    next_state_phi[k] = next_state_phi[k] - 1
                elif(next_state_phi[k]<0):
                    next_state_phi[k] = next_state_phi[k] + 1
                #b.acquire reward
                #寫一個計算reflection_coefficient,transmission_coefficient的function
                #old
                tra = transmission_coefficient(M,state_phi,phi_r_a,state_btea,btea_r_a)
                ref = reflection_coefficient(M,state_phi,phi_r_a,state_btea,btea_r_a)
                cr = H_d + np.matmul(np.matmul(H_vr,ref),H_g)
                ct = np.matmul(np.matmul(H_vt,tra),H_g)
                R_old = DataRate(cr,ct,p1_a[state_p1],p2_a[state_p2],Noise)
                #new
                tra = transmission_coefficient(M,next_state_phi,phi_r_a,state_btea,btea_r_a)
                ref = reflection_coefficient(M,next_state_phi,phi_r_a,state_btea,btea_r_a)
                cr = H_d +  np.matmul(np.matmul(H_vr,ref),H_g)
                ct = np.matmul(np.matmul(H_vt,tra),H_g)
                R_new = DataRate(cr,ct,p1_a[state_p1],p2_a[state_p2],Noise)
                Rt_new = DataRate_t(ct,p1_a[state_p1],p2_a[state_p2],Noise)
                rt = Reward(R_old,R_new,Rt_new)
                #print(rt)
                #Observe next state s_t+1
                max_action = np.argmax(phi_r[k,next_state_phi[k],:])
                #Update Q table
                phi_r[k][state_phi[k]][action] = (1 - alpha) *  phi_r[k][state_phi[k]][action] + alpha * (rt + r * phi_r[k][next_state_phi[k]][max_action])
                #切換到下個state
                state_phi[k] = next_state_phi[k]
        elif(i==3): #調整btea
            for k in range(M):
                #Select action
                if(np.random.rand() < exp*math.exp(-1*t/100)):
                    action = np.random.randint(3)
                else:
                    action = np.argmax(btea_r[k,state_btea[k],:])
                #Perform action and acquire reward
                #a.找出next state(Perform action)
                if(action==0):
                    next_state_btea[k] = state_btea[k] + 1
                elif(action==1):
                    next_state_btea[k] = state_btea[k] - 1
                elif(action==2):
                    next_state_btea[k] = state_btea[k]
                if(next_state_btea[k]>10):
                    next_state_btea[k] = next_state_btea[k] - 1
                elif(next_state_btea[k]<0):
                    next_state_btea[k] = next_state_btea[k] + 1
                #b.acquire reward
                #寫一個計算reflection_coefficient,transmission_coefficient的function
                #old
                tra = transmission_coefficient(M,state_phi,phi_r_a,state_btea,btea_r_a)
                ref = reflection_coefficient(M,state_phi,phi_r_a,state_btea,btea_r_a)
                cr = H_d + np.matmul(np.matmul(H_vr,ref),H_g)
                ct = np.matmul(np.matmul(H_vt,tra),H_g)
                R_old = DataRate(cr,ct,p1_a[state_p1],p2_a[state_p2],Noise)
                #new
                tra = transmission_coefficient(M,state_phi,phi_r_a,next_state_btea,btea_r_a)
                ref = reflection_coefficient(M,state_phi,phi_r_a,next_state_btea,btea_r_a)
                cr = H_d +  np.matmul(np.matmul(H_vr,ref),H_g)
                ct = np.matmul(np.matmul(H_vt,tra),H_g)
                R_new = DataRate(cr,ct,p1_a[state_p1],p2_a[state_p2],Noise)
                Rt_new = DataRate_t(ct,p1_a[state_p1],p2_a[state_p2],Noise)
                rt = Reward(R_old,R_new,Rt_new)
                #print(rt)
                #Observe next state s_t+1
                max_action = np.argmax(btea_r[k,next_state_btea[k],:])
                #Update Q table
                btea_r[k][state_btea[k]][action] = (1 - alpha) *  btea_r[k][state_btea[k]][action] + alpha * (rt + r * btea_r[k][next_state_btea[k]][max_action])
                #切換到下個state
                state_btea[k] = next_state_btea[k]
    tra = transmission_coefficient(M,state_phi,phi_r_a,state_btea,btea_r_a)
    ref = reflection_coefficient(M,state_phi,phi_r_a,state_btea,btea_r_a)
    cr = H_d +  np.matmul(np.matmul(H_vr,ref),H_g)
    ct = np.matmul(np.matmul(H_vt,tra),H_g)
    D_Rate[t] = DataRate(cr,ct,p1_a[state_p1],p2_a[state_p2],Noise)
    D_Rate_Rt[t] = DataRate_t(ct,p1_a[state_p1],p2_a[state_p2],Noise)
    D_Rate_Rr[t] = DataRate_r(cr,p1_a[state_p1],p2_a[state_p2],Noise)
    t = t + 1
    
#Output optimized variable
plt.figure()
plt.plot(moving_average(D_Rate, 10))
plt.title('Multi-agent Q-learning for STAR')
plt.xlabel('epoch')
plt.ylabel('Data Rate')
plt.xlim(0,Tmax)
plt.show()

plt.figure()
plt.plot(moving_average(D_Rate_Rt, 10))
plt.title('Multi-agent Q-learning for STAR')
plt.xlabel('epoch')
plt.ylabel('Data Rate t')
plt.xlim(0,Tmax)
plt.show()

plt.figure()
plt.plot(moving_average(D_Rate_Rr, 10))
plt.title('Multi-agent Q-learning for STAR')
plt.xlabel('epoch')
plt.ylabel('Data Rate r')
plt.xlim(0,Tmax)
plt.show()
