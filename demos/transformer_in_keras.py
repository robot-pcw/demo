#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 1/19/21 3:33 下午
# @Author  : pcw
# @File    : model.py
# @Description: <>
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

"""
Text classification with Transformer
https://keras.io/examples/nlp/text_classification_with_transformer/
"""

class TransformerBlock(layers.Layer):
    """Transformer Encoder: self-attention + residual short-cut + layer normalization"""
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layer_norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layer_norm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layer_norm2(out1 + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, max_len, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        """nonstandard position encode"""
        self.pos_emb = layers.Embedding(input_dim=max_len, output_dim=embed_dim)
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)

    def call(self, x):
        max_len = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=max_len, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


vocab_size = 20000  # Only consider the top 20k words
max_seq_len = 200  # Only consider the first 200 words of each movie review

def prepare_dataset():
    (x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(num_words=vocab_size)
    print(len(x_train), "Training sequences")
    print(len(x_val), "Validation sequences")
    x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_seq_len)
    x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=max_seq_len)
    return (x_train,y_train, x_val,y_val)


def build_classifier_model():
    embed_dim = 32  # Embedding size for each token
    num_heads = 2  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer

    inputs = layers.Input(shape=(max_seq_len,))
    embedding_layer = TokenAndPositionEmbedding(max_seq_len, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(2, activation="softmax")(x)

    return keras.Model(inputs=inputs, outputs=outputs)

if __name__ == '__main__':
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    x_train, y_train, x_val, y_val = prepare_dataset()
    model = build_classifier_model()
    model.summary()
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(
    x_train, y_train, batch_size=32, epochs=2, validation_data=(x_val, y_val))
