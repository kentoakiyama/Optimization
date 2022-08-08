import numpy as np

class GradientDescent:
    def __init__(self, func, der, alpha=0.1, max_iters=100, xi=0.1, tau=0.9, tols=1e-10, min_max=None):
        self.func = func            # 目的関数（最小化）
        self.der = der              # 勾配を計算する関数
        self.alpha = alpha          # 更新幅（Noneの場合は直線探索を使用）
        self.max_iters = max_iters  # 最大更新回数
        self.xi = xi                # アルミ法の条件
        self.tau = tau              # 直線探索時の更新幅の変化量
        self.tols = tols            # 方向微係数の大きさ
        self.min_max = min_max      # 最大値と最小値のタプル, (min_array, max_array)
        
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
        if self.alpha is not None:
            alpha = 5
            while (self.func(self._mod_x(x-alpha*deriv)) > y - self.xi*alpha*np.dot(deriv, deriv)):
                alpha = self.tau * alpha
                if alpha <= 1e-10:
                    break
            return self.alpha
        else:
            return self.alpha
    
    def _print_verbosity(self, step, x, y, deriv):
        x_string = ', '.join([f'{round(i, 5)}'.rjust(7) for i in x])
        deriv_string = ', '.join([f'{round(i, 5)}'.rjust(7) for i in deriv])
        print(f'Step {step: >5}: '+ x_string + f', {round(y, 5): >7}, ' +  deriv_string)

    def minimize(self, x_start, verbosity=0):
        '''
        x_start: 初期解: eg.) np.array([1., 2.])
        '''
        x = np.array(x_start.copy())
        
        for i in range(self.max_iters):
            y = self.func(x)
            deriv = self.der(x)
            
            self.y_history.append(y.copy())
            self.x_history.append(x.copy())
            self.der_history.append(deriv.copy())
            
            if np.linalg.norm(deriv) <= self.tols:
                break
            
            if verbosity != 0:
                if (i+1) % verbosity == 0:
                    self._print_verbosity(i+1, x, y, deriv)

            alpha = self._calc_alpha(x, y, deriv)
            
            x -= alpha*deriv
            x = self._mod_x(x)
        return {'step': i,
                'x': x,
                'y': y,
                'deriv': deriv}
