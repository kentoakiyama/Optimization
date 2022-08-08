import numpy as np
from tqdm import tqdm

class GradientDescent:
    def __init__(self, func, der, alpha=0.1, max_iters=100, xi=0.1, tau=0.9, tols=1e-10, min_max=None, verbosity=0):
        self.func = func            # 目的関数（最小化）
        self.der = der              # 勾配を計算する関数
        self.alpha = alpha          # 更新幅（Noneの場合は直線探索を使用）
        self.max_iters = max_iters  # 最大更新回数
        self.xi = xi                # アルミ法の条件
        self.tau = tau              # 直線探索時の更新幅の変化量
        self.tols = tols            # 方向微係数の大きさ
        self.min_max = min_max      # 最大値と最小値のタプル, (min_array, max_array)
        self.verbosity = verbosity
        
        self.y_history = []
        self.x_history = []
        self.der_history = []
        
    def _mod_x(self, x):
        # min_maxが指定されていたらクリップ、それ以外はそのまま
        if self.min_max is not None:
            return np.clip(x.copy(), self.min_max[0], self.min_max[1])
        else:
            return x
    
    def _calc_alpha(self, x, y, deriv):
        alpha = 5
        while (self.func(self._mod_x(x-alpha*deriv)) > y - self.xi*alpha*np.dot(deriv, deriv)):
            alpha = self.tau * alpha
            if alpha <= 1e-10:
                break
        return alpha
        

    def minimize(self, x_start):
        '''
        x_start: 初期解: eg.) np.array([1., 2.])
        '''
        x = np.array(x_start.copy())
        
        for _ in tqdm(range(self.max_iters)):
            y = self.func(x)
            deriv = self.der(x)
            
            self.y_history.append(y.copy())
            self.x_history.append(x.copy())
            self.der_history.append(deriv.copy())
            
            if np.linalg.norm(deriv) <= self.tols:
                break
            
            if self.alpha is None:
                alpha = self._calc_alpha(x, y, deriv)
            else:
                alpha = self.alpha
            
            x -= alpha*deriv
            x = self._mod_x(x)
