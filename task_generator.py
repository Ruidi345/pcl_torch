import os
import subprocess
import random
import copy
import yaml

class TaskGenerator():
    def __init__(self,):
        with open('./trained_param.yaml') as f:
            trained_cfg = yaml.load(f, Loader=yaml.FullLoader)
        self.base = trained_cfg['tau']
        self.train_dict = {}
        for k, v in trained_cfg.items():
            l = [v.strip() for v in v.split(', ')]
            self.train_dict[k] = l


        self.normal_config = []
        with open('./config.yaml') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        self.normal_config.extend(f"--{k}={v}" for k, v in cfg.items())


        self.repeats = 5
        self.random_max = 1000
        return
    
    def random_seed(self):
        return random.randint(0, self.random_max)
    
    def dupliacte_cmds(self, cmd_set, new_cmd, dup_nums):
        duplicated_cmd_set = []
        for i, cmd in enumerate(cmd_set):
            duplicated_cmd_set.append(copy.deepcopy(new_cmd) for _ in range(dup_nums))   
        return duplicated_cmd_set  
    
    def generate_cmd(self):

        cmd_groups = []
        self.new_cmds = []
        for k, values in self.train_dict.items():

            # don't use new_cmds = [["python","train_f.py"]] * len(values)           
            for i in range(len(values)):
                self.new_cmds.append(["python","trainer.py"])

            for cmd, v in zip(self.new_cmds, values):
                cmd.append(f"--{k}={v}")

        #     print(values)
        #     print(k)
        #     print(cmd)
        #     dup_cmds = self.add_trained_param(cmd, k, values)
        # print(dup_cmds)


    def add_trained_param(self, cmd, key, value):
        """_summary_

        Args:
            cmd (_type_): _description_
            key (str): _description_
            value (list): _description_

        Returns:
            _type_: _description_
        """
        # 先将cmd增列为len(value)个
        dup_cmds = []
        for i in range(len(value)):
            dup_cmds.append(copy.deepcopy(cmd))       
        # print("add_trained_param--: ",dup_cmds)
        # 再对每一个新的dup_cmd添加value
        for cmd, v in zip(dup_cmds, value):
            cmd.append(f"--{key}={v}") 
        # print("add_trained_param: ",dup_cmds)

        return dup_cmds



    def add_seed(self):
        seed_cmd = []
        for cmd in self.new_cmds:
            seed_cmd.extend(copy.deepcopy(cmd) for _ in range(self.repeats))


        for cmd in seed_cmd:
            seed = self.random_seed()
            seed_str = f'--tf_seed={seed}'
            cmd.append(seed_str)

        self.seed_cmd = seed_cmd

        return 0
    
    def add_all_args(self):
        for cmd in self.seed_cmd:
            cmd.extend(self.normal_config)    

  
    def run_all_cmd(self):
        res = []
        for cmd in self.seed_cmd:
            print(cmd)
            res.append(subprocess.call(cmd,shell=False))    
        

    def run(self):
        self.generate_cmd()
        self.add_seed()
        self.add_all_args()
        print(">>>>total nums of tasks:", len(self.seed_cmd),"<<<<")
        self.run_all_cmd()
        return 0


if __name__ == "__main__":
    tg = TaskGenerator()
    tg.run()