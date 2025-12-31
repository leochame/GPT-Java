import com.gpt.TextDataset;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

public class DataPipelineTest {
    public static void main(String[] args) {
        // 1. 假设这是你的输入文件 (为了演示，我直接造个假文件)
        // 实际使用时，请替换为真实路径 "data/input.txt"
        String filePath = "input.txt"; 
        createDummyFile(filePath); 

        // 2. 初始化数据管道
        TextDataset dataset = new TextDataset(filePath);
        
        // 3. 测试编码解码
        String testStr = "山路";
        int[] encoded = dataset.encode(testStr);
        System.out.println("---------------------------------");
        System.out.println("测试编码: '" + testStr + "' -> " + Arrays.toString(encoded));
        System.out.println("测试解码: " + Arrays.toString(encoded) + " -> '" + dataset.decode(encoded) + "'");
        
        // 4. ★核心★：测试 Batch 获取
        // 假设我们一次取 2 条数据 (BatchSize=2)，每条数据看 4 个字 (BlockSize=4)
        System.out.println("---------------------------------");
        System.out.println("正在抽取训练 Batch...");
        TextDataset.BatchData batch = dataset.getBatch(2, 4);
        
        for (int i = 0; i < 2; i++) {
            System.out.println("\n样本 " + i + ":");
            System.out.println("  输入 (x) ID: " + Arrays.toString(batch.inputs[i]));
            System.out.println("  目标 (y) ID: " + Arrays.toString(batch.targets[i]));
            
            // 把 ID 翻译回文字，看看逻辑对不对
            System.out.println("  [可视化]");
            System.out.println("  输入: \"" + dataset.decode(batch.inputs[i]) + "\"");
            System.out.println("  目标: \"" + dataset.decode(batch.targets[i]) + "\"");
        }
    }

    // 辅助方法：生成一个测试文件
    private static void createDummyFile(String path) {
        try {
            String content = "这里的山路十八弯，这里的水路九连环。没有比人更高的山，没有比脚更长的路。";
//            Files Files = "input.txt";
            Files.writeString(Path.of(path), content);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}