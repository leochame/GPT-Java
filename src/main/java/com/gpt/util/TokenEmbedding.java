package com.gpt.util;

import java.util.Random;
import java.util.Arrays;

public class TokenEmbedding {
    // 权重矩阵: [vocab_size, output_dim]
    // 对应书中的 embedding_layer.weight
    public double[][] weight;

    /**
     * 构造函数：创建嵌入层
     * 对应书中代码: torch.nn.Embedding(vocab_size, output_dim)
     */
    public TokenEmbedding(int vocabSize, int embDim) {
        // 1. 分配空间
        this.weight = new double[vocabSize][embDim];
        
        // 2. 初始化权重 (Initialize weights)
        // 书中使用 torch.manual_seed(123)
        // 这里我们用 Java 的 Random(123) 确保生成的随机数和书中逻辑一致
        Random random = new Random(123);
        
        for (int i = 0; i < vocabSize; i++) {
            for (int j = 0; j < embDim; j++) {
                // 使用较小的随机值初始化
                this.weight[i][j] = random.nextGaussian() * 0.02; 
            }
        }
    }

    // 辅助方法：打印权重矩阵 (用于验证)
    // 对应书中: print(embedding_layer.weight)
    public void printWeights() {
        System.out.println("Parameter containing:");
        System.out.print("tensor([");
        for (int i = 0; i < weight.length; i++) {
            System.out.print("[");
            for (int j = 0; j < weight[i].length; j++) {
                System.out.printf("%8.4f", weight[i][j]);
                if (j < weight[i].length - 1) System.out.print(",");
            }
            System.out.print("]");
            if (i < weight.length - 1) System.out.println(",");
        }
        System.out.println("], requires_grad=True)");
    }

    public static void main(String[] args) {
        // 1. 定义参数 (对应书中示例)
        int vocabSize = 6;
        int outputDim = 3;

        // 2. 实例化嵌入层
        // 对应代码: embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
        TokenEmbedding embeddingLayer = new TokenEmbedding(vocabSize, outputDim);

        // 3. 查看权重
        // 对应代码: print(embedding_layer.weight)
        embeddingLayer.printWeights();

    }
}