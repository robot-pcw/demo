#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 1/19/21 11:11 上午
# @Author  : pcw
# @File    : layers.py
# @Description: <build Transform in tf.keras>
import tensorflow as tf
import tensorflow.keras as k
import tensorflow.keras.layers as kl


class EncoderLayer(kl.Layer):
    """Transformer Encoder Block"""
    def __init__(self, key_dim, num_heads, dropout_rate=0.0, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        self.self_attention = MultiHeadAttention(self.key_dim, self.num_heads)
        self.attention_res_norm = AddNormLayer(dropout_rate)

        self.feed_forward = kl.Dense(self.key_dim, activation='relu')
        self.ff_res_norm = AddNormLayer(dropout_rate)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x, **kwargs):
        z,attention = self.self_attention(query=x, value=x, key=x, masks=None, return_attention=True)
        z = self.attention_res_norm([x, z])
        ff = self.feed_forward(z)
        res = self.ff_res_norm([z, ff])
        return (res, attention)

    def get_config(self):
        config = {
            'key_dim': self.key_dim,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DecoderLayer(kl.Layer):
    """Transformer Decoder Block"""
    def __init__(self, key_dim, num_heads, dropout_rate, **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        self.self_attention = MultiHeadAttention(self.key_dim, self.num_heads)
        self.self_attention_res = AddNormLayer(dropout_rate)

        self.encoder_decoder_attention = MultiHeadAttention(self.key_dim, self.num_heads)
        self.coder_attention_res = AddNormLayer(dropout_rate)

        self.feed_forward = kl.Dense(self.key_dim, activation='relu')
        self.ff_res = AddNormLayer(dropout_rate)

    def call(self, inputs, encdoer_output, later_mask, padding_mask):
        z1,attention1 = self.self_attention(query=inputs, value=inputs, key=inputs,
                                            masks=later_mask, return_attention=True)
        z1 = self.self_attention_res([inputs, z1])

        z2,attention2 = self.encoder_decoder_attention(query=z1, value=encdoer_output, key=encdoer_output,
                                                       return_attention=True)
        z2 = self.coder_attention_res([z1, z2])

        z3 = self.feed_forward(z2)
        z3 = self.ff_res([z2, z3])
        return (z3, attention1, attention2)


    def get_config(self):
        config = {
            'key_dim': self.key_dim,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AddNormLayer(kl.Layer):
    """wrap layers with residual, normalization and dropout."""
    def __init__(self, dropout_rate=0.0, **kwargs):
        super(AddNormLayer, self).__init__(**kwargs)
        self.dropout_rate = dropout_rate
        self.dropout = None

    def build(self, input_shape):
        if self.dropout_rate > 0.0:
            self.dropout = kl.Dropout(self.dropout_rate)
        self.res_short_cut = k.layers.Add()
        self.layer_norm = kl.LayerNormalization()
        super(AddNormLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        res, x = inputs
        x = x if self.dropout is None else self.dropout(x)
        x = self.res_short_cut([res, x])
        x = self.layer_norm(x)
        return x


class MultiHeadAttention(kl.Layer):
    """
    Examples:
    Performs 1D cross-attention over two sequence inputs with an attention mask.
    Returns the additional attention weights over heads.
    >>> layer = MultiHeadAttention(num_heads=2, key_dim=2)
    >>> target = tf.keras.Input(shape=[8, 16])
    >>> source = tf.keras.Input(shape=[4, 16])
    >>> output_tensor, weights = layer(target, source,
    ...                                return_attention_scores=True)
    >>> print(output_tensor.shape)
    (None, 8, 16)
    >>> print(weights.shape)
    (None, 2, 8, 4)
    """

    def __init__(self, key_dim, num_heads, output_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.k_dim = key_dim
        self.n_heads = num_heads
        self.output_dim = output_dim
        self.qw = kl.Dense(self.k_dim * self.n_heads, use_bias=True)
        self.kw = kl.Dense(self.k_dim * self.n_heads, use_bias=True)
        self.vw = kl.Dense(self.k_dim * self.n_heads, use_bias=True)

    def call(self, query, value, key=None, masks=None, return_attention=False):
        # 将输入中的embedding向量仿射为Query，Key，Value矩阵
        if key is None:
            key = value
        out_dim = query.shape[-1] if self.output_dim is None else self.output_dim
        self.multi_dense = kl.Dense(out_dim)
        q = self.qw(query)
        k = self.kw(key)
        v = self.vw(value) # (..., k_dim*n_heads)

        queries = split_into_heads(q, self.n_heads) # (..., n_heads, seq_len,  k_dim)
        keys = split_into_heads(k, self.n_heads)
        values = split_into_heads(v, self.n_heads)
        attention, mha_values = self._apply_attention_to_values(queries, keys, values, masks)

        # 多头合并输出
        _, n_heads, q_len, d_head = mha_values.shape
        tf.transpose(mha_values,  perm=[0,2,1,3])
        concat = tf.reshape(mha_values, [-1, q_len, n_heads*d_head])
        result = self.multi_dense(concat)
        return (result, attention) if return_attention else result

    def _apply_attention_to_values(self, q, k, v, masks):
        """
        q为查询向量，表示为了编码当前词需要去注意(attend to)其它(含自身)的词；
        k向量可以认为是这个词被用于被检索的表示向量；
        v向量是该词需要结合上下文被编码的内容。
        :param q: (b_size, n_heads, q_len, k_dim)
        :param k: (b_size, n_heads, k_len, k_dim)
        :param v: (b_size, n_heads, k_len, k_dim)
        :param masks: (b_size, n_heads, q_len, k_len)
        :return: (b_size, n_heads, q_len, k_dim)
        """
        score = dot_product_score(q, k)
        d = k.shape[-1]
        scale = tf.constant(d, dtype=tf.float32, name="scale")
        scale_score = score / tf.sqrt(scale)
        if masks:
            # mask指定位置的score值
            scale_score -= 1.e9 * masks
        attention = tf.nn.softmax(scale_score, axis=-1)  # (b_size, n_heads, q_len, k_len)
        return attention, tf.matmul(attention, v)

    def get_config(self):
        config = {
            'd': self.k_dim,
            'heads': self.n_heads,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


def dot_product_score(q: tf.Tensor, k: tf.Tensor):
    """
    :param q: tensor with shape (b_size, n_heads, q_len, d_head)
    :param k: tensor with shape (b_size, n_heads, k_len, d_head)
    :return:  tensor with shape (b_size, n_heads, q_len, k_len)
    """
    return tf.matmul(q, k, transpose_b=True)


def split_into_heads(tensor: tf.Tensor, heads) -> tf.Tensor:
    """
    :param tensor: (b_size, seq_len, d_feature)
    :param heads: (b_size, seq_len, d_feature)
    :return: (b_size, n_heads, seq_len, d_head)
    """
    _, seq_len, d_features = tensor.shape
    if d_features % heads != 0:
        raise ValueError("d_feature / d_head error!")
    d_head = d_features // heads
    tensor_split = tf.reshape(tensor, [-1, seq_len, heads, d_head])
    return tf.transpose(tensor_split, perm=[0,2,1,3])


if __name__ == '__main__':
    #mha = MultiHeadAttention(key_dim=32, num_heads=2)
    mha = EncoderLayer(key_dim=64, num_heads=2)
    input_tensor = tf.keras.Input(shape=[100, 64])
    output_tensor = mha(input_tensor)
    print(output_tensor.shape)




