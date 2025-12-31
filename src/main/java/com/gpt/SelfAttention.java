package com.gpt;

import com.gpt.util.MatrixUtils;

import java.util.Arrays;
import java.util.Random;

public class SelfAttention {
    private int dModel; // 输入向量维度 (例如 64)
    private int dK;     // Q, K, V 的内部维度 (例如 64)

    // 三个可学习的权重矩阵
    // 它们决定了模型如何提取特征
    private double[][] W_Q;
    private double[][] W_K;
    private double[][] W_V;

    public SelfAttention(int dModel) {
        this.dModel = dModel;
        this.dK = dModel; // 简化起见，让内部维度等于输入维度

        // 随机初始化权重
        this.W_Q = randomMatrix(dModel, dK);
        this.W_K = randomMatrix(dModel, dK);
        this.W_V = randomMatrix(dModel, dK);
    }

    // 前向传播：Attention 的核心流程
    // 输入 x: [序列长度, dModel] -> 比如 [3, 64]
    public double[][] forward(double[][] x) {
        
        // 1. 线性投影 (Linear Projections)
        // 公式: Q = X * W_Q, K = X * W_K, V = X * W_V
        // 这里的物理意义：把原始向量映射到“查询空间”、“键空间”和“值空间”
        double[][] Q = MatrixUtils.matmul(x, W_Q);
        double[][] K = MatrixUtils.matmul(x, W_K);
        double[][] V = MatrixUtils.matmul(x, W_V);

        // 2. 计算注意力分数 (Scaled Dot-Product Attention)
        // 公式: Scores = Q * K^T
        // 物理意义：计算每个词和其他词的相似度 (点积越大越相似)
        double[][] K_T = MatrixUtils.transpose(K);
        double[][] scores = MatrixUtils.matmul(Q, K_T);
        
        // 形状变化: [3, 64] * [64, 3] = [3, 3] (这是一个 N*N 的方阵)
        // scores[0][1] 代表第0个词(我)对第1个词(爱)的关注度

        // 3. 缩放 (Scaling)
        // 公式: Scores / sqrt(dK)
        // 作用：防止点积结果太大，导致 Softmax 梯度消失
        scale(scores, Math.sqrt(dK));

        // 4. 掩码 (Masking) - GPT 的关键！
        // 作用：把矩阵右上角变成负无穷，确保“我”看不见“爱”和“Java”
        applyCausalMask(scores);

        // 5. 归一化 (Softmax)
        // 作用：把分数变成概率 (0.0 ~ 1.0)
        double[][] attentionWeights = MatrixUtils.softmax(scores);
        
        // 打印一下权重看看 (调试用)
        System.out.println("注意力权重矩阵 (Softmax后):");
        printMatrix(attentionWeights);

        // 6. 加权求和 (Weighted Sum)
        // 公式: Output = Weights * V
        // 物理意义：根据刚才算出的概率，混合 V 中的信息
        // 比如：0.7 * (我的V) + 0.3 * (爱的V)
        double[][] output = MatrixUtils.matmul(attentionWeights, V);

        return output;
    }

    // 辅助：生成随机矩阵
    private double[][] randomMatrix(int rows, int cols) {
        double[][] m = new double[rows][cols];
        Random r = new Random();
        for(int i=0;i<rows;i++) 
            for(int j=0;j<cols;j++) 
                m[i][j] = r.nextGaussian() * 0.02;
        return m;
    }
    
    // 辅助：除以常数
    private void scale(double[][] matrix, double factor) {
        for(int i=0;i<matrix.length;i++)
            for(int j=0;j<matrix[0].length;j++)
                matrix[i][j] /= factor;
    }

    // 辅助：因果掩码 (Causal Mask)
    // 也就是下三角矩阵
    private void applyCausalMask(double[][] scores) {
        for (int i = 0; i < scores.length; i++) {
            for (int j = 0; j < scores[0].length; j++) {
                // 如果 j > i (列号 > 行号)，说明是“未来的词”
                if (j > i) {
                    scores[i][j] = -1e9; // 负无穷大
                }
            }
        }
    }
    
    // 调试打印
    private void printMatrix(double[][] m) {
        for (double[] row : m) {
            System.out.println(Arrays.toString(row));
        }
    }
}