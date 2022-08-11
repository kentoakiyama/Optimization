# 勾配ベースの最適化をまとめています

## 最適化リスト
- [Gradient Descent](#gradient-descent)
- [Newton method (in future)](#newton-method)
- [SQP method (in future)](#sqp-method)
- [Dynamic programing (in future)](#dynamic-programing)

## Gradient Descent
最急降下法を用いた最適化になります。

サンプルコードをexamples/gradient_descent_eg.pyに示しているので御覧ください。

簡単に使い方を説明しますと、

```python
gd = GradientDescent(func, deriv, alpha, max_iters, ...)
result = gd.minimize(x_start, verbosity)
```

で動かします。

GradientDescentの主な引数
- func: 目的関数（最小化）
- deriv: 目的関数の微分
- alpha: 更新幅（Noneに設定すると直線探索を用いてalphaを決定します）
- max_iters: 最大更新回数

minimizeの引数
- x_start: 初期解
- verbosity: ログ出力の頻度

それ以外の引数についてはソースコードをご覧ください。


結果は、
- 返り値(return) -> {'step': 最終ステップ数, 'x': 座標値, 'y': 目的関数の値, 'deriv': 方向微係数}
- gd.y_history -> 各ステップの目的関数のリスト
- gd.x_history -> 各ステップの座標のリスト
- gd.der_history -> 各ステップの方向微係数のリスト

で取得できます。


## Newton method

mainで作成されている

## SQP method

## Dynamic programing
これはdevelopでさξされている

# 参考文献
## Gradient descent
- [直線探索を使った最急降下法をPythonで実装](https://helve-blog.com/posts/math/gradient-descent-armijo/#:~:text=%E7%9B%B4%E7%B7%9A%E6%8E%A2%E7%B4%A2%20(line%20search)%20%E3%81%AF,%E8%A8%98%E4%BA%8B%E3%81%A7%E3%81%AF%E5%89%8D%E8%80%85%E3%81%AE%E3%81%BF%E6%89%B1%E3%81%86%E3%80%82)