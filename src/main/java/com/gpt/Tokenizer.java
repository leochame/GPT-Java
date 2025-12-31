package com.gpt;

import java.util.*;

public class Tokenizer {
    //  (Padding, ID 0): 填充标记。
    public static final String STR_PAD = "[PAD]";
    // (Unknown, ID 1): 未知词标记
    public static final String STR_UNK = "[UNK]";
    // (Begin of Sentence, ID 2): 句首标记
    public static final String STR_BOS = "[BOS]";
    // (End of Sentence, ID 3): 句尾标记
    public static final String STR_EOS = "[EOS]";

    public static final int ID_PAD = 0;
    public static final int ID_UNK = 1;
    public static final int ID_BOS = 2;
    public static final int ID_EOS = 3;

    private Map<Character, Integer> charToId = new HashMap<>();
    private Map<Integer, String> idToChar = new HashMap<>(); // 注意：这里Value变成String，为了兼容特殊标记
    private int vocabSize;

    public Tokenizer(String rawText) {
        // 1. 初始化特殊标记
        initSpecialTokens();
        // 2. 构建词表
        buildVocabulary(rawText);
    }

    private void initSpecialTokens() {
        // 严格按照 ID 顺序压入
        idToChar.put(ID_PAD, STR_PAD);
        idToChar.put(ID_UNK, STR_UNK);
        idToChar.put(ID_BOS, STR_BOS);
        idToChar.put(ID_EOS, STR_EOS);
    }

    private void buildVocabulary(String text) {
        // 去重并排序，确保确定性
        Set<Character> chars = new HashSet<>();
        for (char c : text.toCharArray()) {
            chars.add(c);
        }
        List<Character> sortedChars = new ArrayList<>(chars);
        Collections.sort(sortedChars);

        // 普通字符从 ID 4 开始（因为 0,1,2,3 被占用了）
        int currentId = 4;
        for (char c : sortedChars) {
            charToId.put(c, currentId);
            idToChar.put(currentId, String.valueOf(c));
            currentId++;
        }
        this.vocabSize = currentId;
    }

    /**
     * 编码逻辑 (符合 GPT 生成任务的标准流程)
     * 输入: "你好"
     * 输出: [BOS, 你, 好, EOS]
     * * @param text 原始文本
     * @param maxLen 强制最大长度 (用于 Padding)
     */
    public int[] encode(String text, int maxLen) {
        List<Integer> ids = new ArrayList<>();

        // 1. 插入句首标记 [BOS]
        ids.add(ID_BOS);

        // 2. 转换文本内容
        for (char c : text.toCharArray()) {
            ids.add(charToId.getOrDefault(c, ID_UNK));
        }

        // 3. 插入句尾标记 [EOS]
        ids.add(ID_EOS);

        // 4. 截断 (Truncation) 或 填充 (Padding)
        int[] result = new int[maxLen];
        for (int i = 0; i < maxLen; i++) {
            if (i < ids.size()) {
                result[i] = ids.get(i);
            } else {
                // 长度不足，补 [PAD]
                result[i] = ID_PAD;
            }
        }
        return result;
    }

    /**
     * 解码逻辑
     * 遇到 [EOS] 停止，忽略 [PAD] 和 [BOS]
     */
    public String decode(int[] ids) {
        StringBuilder sb = new StringBuilder();
        for (int id : ids) {
            if (id == ID_BOS) continue; // 跳过句首
            if (id == ID_PAD) continue; // 跳过填充
            if (id == ID_EOS) break;    // ★ 遇到结束符必须立即停止

            sb.append(idToChar.getOrDefault(id, STR_UNK));
        }
        return sb.toString();
    }

    public int getVocabSize() { return vocabSize; }
}