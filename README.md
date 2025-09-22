For DQN

# "o2" is the parameter for calling the true oracle
# "max_del" is not using it right now, ignore
# "step" is the parameter of using multiple steps of reward, pass 1 to disable multi-steps

python deep_q_learning.py --budget 16 --o2 100 --lr 1e-4 --max_del 4 --steps 3 --device 0   