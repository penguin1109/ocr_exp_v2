import math
from torch.optim.lr_scheduler import _LRScheduler
"""
- 이걸 사용하기 위해서는 optimizer의 시작 learning rate는 0, 혹은 0에 가까운 아주 작은 수를 입ㄹ력 해야 한다.
- eta_max: learning rate의 최대값
- T_0: 최초 주기값
- T_mult: 주기가 반복되면서 최초 주기에 비해서 얼마나 늘릴것인지에 대한 scale
- eta_min: learning rate의 최소값
- T_up: Warmup을 할 때 필요하느 epoch의 수
- gamma: 주기가 반복될수록 eta_max에 곱해지는 실수 값이다.(만약에 gamma가 1 미만이라면 주기가 반복될 때마다 learning rate의 최대 값이 작아진다.)
"""
class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected Positive integer T_0, but got { }".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected Positive Integer T_mult, but got { }".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected Positive Integer T_up, but got { }".format(T_up))
        
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    def get_lr(self):
        if self.T_cur == -1: ## 현재 time step이 처음 단계라면
            return self.base_lrs 
        elif self.T_cur < self.T_up: ## learning rate를 일정 크기 올려야 하는 step이 아직 아닌 경우에
            return [(self.eta_max - base_lr) * self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur - self.T_up) / (self.T_i - self.T_up))) for base_lr in self.base_lrs]
    
    def step(self, epoch = None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 + self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
        self.eta_max = self.base_eta_max * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr ## 현재 이 scheduler이 관리하고 있는 optimizer의 learning rate값을 변경 해 준다.
