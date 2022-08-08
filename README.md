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

## SQP method

## Dynamic programing