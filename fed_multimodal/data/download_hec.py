import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

def generate_samples(a, b, A, B, N):
  """
  生成不同 Asin(ax + b) + B 的三角函数采样点

  Args:
    a: 三角函数的1/频率
    b: 三角函数的水平偏移
    A: 三角函数的垂直偏移
    B: 三角函数的纵轴截距
    N: 采样点数

  Returns:
    采样点数组
  """
  # 生成采样时间点
  t = np.linspace(0, 2 * np.pi, N)
  # 生成采样点
  x = A * np.sin(a * t + b) + B
  return x

output_data_path = Path(os.path.realpath(__file__)).parents[0].joinpath('hec')
sampling_rate = 30

def hec():
  """
  数据项左到右分别为:振幅f1,f2,f3,温度t1,t2,t3,
  """
  data = np.zeros(shape=[50*1000*30,8])
  for num in tqdm(range(50)):
      count = 0
      x = np.random.randint(1, 7, size=[1000])
      y = np.zeros(shape=[30, 8])
      A = np.zeros(shape=[6])
      B = np.zeros(shape=[6])
      a = np.zeros(shape=[6])
      b = np.zeros(shape=[6])
      for idx in x:
          if idx==1 : # 差
              A[:3] = np.random.uniform(10, 5, 3)
              B[:3] = np.zeros(shape=[3])
              a[:3] = np.random.uniform(5, 15, 3)
              b[:3] = np.random.uniform(0, 3.14, 3)
              A[3:] = np.ones(shape=3) * 2
              B[3:] = np.random.uniform(80, 100, 3)
              a[3:] = np.random.uniform(5, 15, 3)
              b[3:] = np.random.uniform(0, 3.14, 3)
              y[:,6] = np.ones(shape=[30]) * 0
          elif idx<=4 : # 一般
              A[:3] = np.random.uniform(5, 10, 3)
              B[:3] = np.zeros(shape=[3])
              a[:3] = np.random.uniform(5, 15, 3)
              b[:3] = np.random.uniform(0, 3.14, 3)
              A[3:] = np.ones(shape=[3]) * 2
              B[3:] = np.random.uniform(40, 80, 3)
              a[3:] = np.random.uniform(5, 15, 3)
              b[3:] = np.random.uniform(0, 3.14, 3)
              y[:,6] = np.ones(shape=[30]) * 1
          else :      # 良好
              A[:3] = np.random.uniform(0, 5, 3)
              B[:3] = np.zeros(shape=[3])
              a[:3] = np.random.uniform(5, 15, 3)
              b[:3] = np.random.uniform(0, 3.14, 3)
              A[3:] = np.ones(shape=[3]) * 2
              B[3:] = np.random.uniform(10, 40, 3)
              a[3:] = np.random.uniform(5, 15, 3)
              b[3:] = np.random.uniform(0, 3.14, 3)
              y[:,6] = np.ones(shape=[30]) * 2
          
          for i in range(6):
              y[:,i+1] = generate_samples(a=a[i], b=b[i], A=A[i], B=B[i], N=sampling_rate)
          y[:, 0] = num+1
          data[num*30000+30*count:num*30000+30*(count+1),:] = y
          count += 1
    
  df = pd.DataFrame({'TurbID':data[:,0],
                     'f1':data[:,1], 
                     'f2':data[:,2], 
                     'f3':data[:,3], 
                     't1':data[:,4], 
                     't2':data[:,5], 
                     't3':data[:,6], 
                     'label':data[:,7]})
  df['TurbID'] = df['TurbID'].astype('int')
  df['label'] = df['label'].astype('int')
  df.to_csv(output_data_path.joinpath("hec.csv"), sep=',', index=False)
  

if __name__ == '__main__':
  print(output_data_path)
  Path.mkdir(output_data_path, parents=True, exist_ok=True)
  hec()