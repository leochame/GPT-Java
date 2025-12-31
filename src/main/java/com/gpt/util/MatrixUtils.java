package com.gpt.util;

import java.util.Arrays;

public class MatrixUtils {

    // 1. 矩阵乘法 (A x B)
    // 对应公式中的 Q * K^T 和 Score * V
    public static double[][] matmul(double[][] A, double[][] B) {
        int rowsA = A.length;
        int colsA = A[0].length;
        int rowsB = B.length;
        int colsB = B[0].length;

        if (colsA != rowsB) throw new IllegalArgumentException("维度不匹配，无法相乘");

        double[][] result = new double[rowsA][colsB];
        // 朴素的矩阵乘法 O(N^3)
        for (int i = 0; i < rowsA; i++) {
            for (int j = 0; j < colsB; j++) {
                for (int k = 0; k < colsA; k++) {
                    result[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return result;
    }

    // 2. 矩阵转置 (T)
    // 对应公式中的 K^T
    public static double[][] transpose(double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[][] result = new double[cols][rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[j][i] = matrix[i][j];
            }
        }
        return result;
    }

    // 3. Softmax (对每一行做归一化)
    // 作用：把分数变成概率，让一行加起来等于 1
    public static double[][] softmax(double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[][] result = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            double max = Double.NEGATIVE_INFINITY;
            // 找最大值 (为了数值稳定性，防止 exp 爆炸)
            for (int j = 0; j < cols; j++) max = Math.max(max, matrix[i][j]);

            double sum = 0.0;
            for (int j = 0; j < cols; j++) {
                double val = Math.exp(matrix[i][j] - max); // 减去最大值
                result[i][j] = val;
                sum += val;
            }

            // 归一化
            for (int j = 0; j < cols; j++) {
                result[i][j] /= sum;
            }
        }
        return result;
    }
}