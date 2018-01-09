Need: Pytorch 0.3, Python3.5+

如果PolicyGradient版本运行结果有异常， 将本项目的reversi/中的reversi.py覆盖掉reversi的原始版本(在gym库的对应位置)。原始的reversi.py中有一点小Bug，会导致Policy Gradient训练不出来。

以上改动不会影响gym的正常运行。

model是已经训练好的模型（训练轮数为1000轮），默认设置黑棋为agent，与随机策略的白棋博弈胜率在80%附近，因为抽样策略的缘故会有些许波动。(理论上胜率会随着训练轮数的增长而提升一些，并且更稳定)。

调用训练好的模型时，将RL_QG_agent的train参数设为False.

如果想自己训练模型，将train设为True即可。模型默认保存为'model'.所以会覆盖掉已存在的model。可在RL_QG_agent源码中修改。


运行：(调用已训练好model)

python3 reveersi_main.py


预期输出：

################################################################

Load model successfully!

Episode 100 done

模型胜利次数：82	总次数：100

模型胜率：0.82

