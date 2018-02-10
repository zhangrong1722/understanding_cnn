**Note：**

- In the code written by `TF`,targeted `Tensor`(e.g. output of model) is saved to collections(e.i. `tf.add_to_collection()`) when training and get them from collections.However,I save the model weights,then load weights and finally predict same Tensor(e.g. output of model) to obtain targeted `Tensor` using pytorch.

- the function `backprop_dense` in `model.py` is implemented on these formula:

  <div align="center">$$R^{(l,l+1)}_{j\gets k}=R^{l+1}_{k}·(\alpha·\frac{z^+_{jk}}{z^+_k}-\beta·\frac{z^-_{jk}}{z^-_k})$$</div>

  <div align="center">$$\beta-\alpha=1$$</div>

  That means the way $k$th neural in $(l+1)$ layer backprop relevance scores to $j$th neural in $l$ layer.

- Let the neurons of the DNN be described by the equation:

  <div align="center">$$a_k=\sigma(\sum_ja_jw_{jk} + b)$$</div>

  <div align="center">$a=\sigma(xw+b)$</div>

  ​

  ​