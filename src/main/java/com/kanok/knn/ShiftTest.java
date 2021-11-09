package com.kanok.knn;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;
import java.util.TreeMap;

public class ShiftTest {

    public void test() {
        //vectors count
        int nodeCount = 1000000;
        //query count
        int qSize = 1000;
        //vector dimension
        int vecDim = 128;
        //number of nearest neighbors to find
        int k = 10;


        /////////////////////////////////////////////////////// READ DATA
        FloatBuffer mass = createFloatBuffer("c:/All/VSB/2rocnik/AVD/sift1M/sift1M.bin");
        FloatBuffer massQ = createFloatBuffer("c:/All/VSB/2rocnik/AVD/sift1M/siftQ1M.bin");
        //IntBuffer massQA = createIntBuffer("c:/All/VSB/2rocnik/AVD/sift1M/knnQA1M.bin");
        /////////////////////////////////////////////////////// QUERY PART

        System.out.println("Start querying");

        int sum = 0;
        for (int t = 0; t < 3; t++) {
            List<Integer> result = new ArrayList<>();
            long start = System.currentTimeMillis();

            for (int i = 0; i < qSize; i++) {
                float[] massQArray = new float[129];
                massQ.position(i * vecDim);
                massQ.get(massQArray, 0, vecDim);
                bruteForce(mass, massQArray, result, vecDim, k, nodeCount);
                System.out.println("For query number: " + i);

            }
            long end = System.currentTimeMillis();
            long time = end - start;
            sum += time;

            result.clear();
        }
        System.out.println("avg: " + (float) sum / (qSize * 3) + " [ms]");
    }

    private void bruteForce(FloatBuffer data, float[] query, List<Integer> result, int dim, int k, int nodeCount) {
        double diff = 0;
        int qI = 0;
        TreeMap<Double, Integer> diffs = new TreeMap<>();
        int size = 0;

        for (int i = 0; i < nodeCount * dim; i++) {
            float tmp = data.get(i) - query[qI];
            diff += (tmp) * (tmp);
            qI++;
            if (qI == dim) {
                diff = Math.sqrt(diff);
                if (size < k) {
                    diffs.put(diff, size);
                } else {
                    double lastKey = diffs.lastKey();
                    if (lastKey > diff) {
                        diffs.remove(lastKey);
                        diffs.put(diff, size);
                    }
                }
                size++;
                qI = 0;
                diff = 0;
            }
        }

        int l = 0;
        for (int val : diffs.values()) {
            System.out.println("K: " + l + " position: " + val);
            l++;
            result.add(val);
        }
    }

    private FloatBuffer createFloatBuffer(String filePath) {
        FloatBuffer mass = null;
        try (FileChannel fc = new RandomAccessFile(filePath, "rw").getChannel()) {
            mass = fc.map(FileChannel.MapMode.READ_WRITE, 0, fc.size())
                    .order(ByteOrder.nativeOrder()).asFloatBuffer();

        } catch (IOException e) {
            e.printStackTrace();
        }
        return mass;
    }

    /*private IntBuffer createIntBuffer(String filePath) {
        IntBuffer intBuffer = null;
        try (FileInputStream stream = new FileInputStream(filePath)) {
            FileChannel inChannel = stream.getChannel();

            ByteBuffer buffer = inChannel.map(FileChannel.MapMode.READ_ONLY, 0, inChannel.size());

            intBuffer = buffer.asIntBuffer();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return intBuffer;
    }*/
}
