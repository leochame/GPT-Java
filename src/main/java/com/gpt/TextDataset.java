package com.gpt;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.stream.Collectors;

public class TextDataset {
    // 原始数据
    private String rawData;
    // 词表大小
    private int vocabSize;
    
    // 核心映射表
    private Map<Character, Integer> charToId = new HashMap<>();
    private Map<Integer, Character> idToChar = new HashMap<>();
    
    // 数字化后的全部数据 (这就变成了巨大的 int 数组)
    private int[] data;

    // 构造函数：传入文件路径
    public TextDataset(String filePath) {
        try {
            System.out.println("正在读取文件...");
            this.rawData = Files.readString(Path.of(filePath));
            
            System.out.println("正在构建词表...");
            buildVocabulary(this.rawData);
            
            System.out.println("正在将文本数字化...");
            this.data = encode(this.rawData);
            
            System.out.println("数据处理完成！");
            System.out.println("文本总长度: " + data.length);
            System.out.println("词表大小: " + vocabSize);
            
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // 1. 构建词表
    private void buildVocabulary(String text) {
        // 使用 Set 去重，获取所有出现过的字符
        Set<Character> chars = new HashSet<>();
        for (char c : text.toCharArray()) {
            chars.add(c);
        }

        // 排序，保证每次运行 ID 都是一样的 (为了可复现性)
        List<Character> sortedChars = new ArrayList<>(chars);
        Collections.sort(sortedChars);

        this.vocabSize = sortedChars.size();

        // 建立双向映射
        for (int i = 0; i < sortedChars.size(); i++) {
            char c = sortedChars.get(i);
            charToId.put(c, i);
            idToChar.put(i, c);
        }
    }

    // 2. 编码: String -> int[]
    public int[] encode(String text) {
        int[] ids = new int[text.length()];
        for (int i = 0; i < text.length(); i++) {
            Character c = text.charAt(i);
            // 如果遇到没见过的字（比如生僻字），这里简单处理为 0 或者抛异常
            // 这里我们假设训练集够大，涵盖了所有字
            if (charToId.containsKey(c)) {
                ids[i] = charToId.get(c);
            } else {
                System.err.println("警告: 未知字符 " + c);
                ids[i] = 0; 
            }
        }
        return ids;
    }

    // 3. 解码: int[] -> String (用于模型输出结果时)
    public String decode(int[] ids) {
        StringBuilder sb = new StringBuilder();
        for (int id : ids) {
            sb.append(idToChar.getOrDefault(id, '?'));
        }
        return sb.toString();
    }
    
    // Getter
    public int getVocabSize() { return vocabSize; }
    public int getDataSize() { return data.length; }

    /**
     * 获取一个批次的训练数据
     * @param batchSize 一次训练多少句话 (并行度)
     * @param blockSize 上下文窗口大小 (模型能看多长)
     * @return 返回两个二维数组: inputs[batch][block], targets[batch][block]
     */
    public BatchData getBatch(int batchSize, int blockSize) {
        int[][] inputs = new int[batchSize][blockSize];
        int[][] targets = new int[batchSize][blockSize];

        Random random = new Random();

        for (int i = 0; i < batchSize; i++) {
            // 随机选一个起点
            // 注意范围：不能选到最后 blockSize 个，否则 target 会越界
            int startIndex = random.nextInt(this.data.length - blockSize - 1);

            // 截取数据
            for (int j = 0; j < blockSize; j++) {
                inputs[i][j] = this.data[startIndex + j];     // 输入
                targets[i][j] = this.data[startIndex + j + 1]; // 目标 (向后错一位)
            }
        }

        return new BatchData(inputs, targets);
    }

    // 简单的数据载体类
    public static class BatchData {
        public int[][] inputs;
        public int[][] targets;

        public BatchData(int[][] x, int[][] y) {
            this.inputs = x;
            this.targets = y;
        }
    }
}