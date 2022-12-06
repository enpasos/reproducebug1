package com.enpasos.bugs;

import ai.djl.basicdataset.cv.classification.Mnist;
import ai.djl.basicmodelzoo.basic.Mlp;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.LayerNorm;
import ai.djl.nn.pooling.Pool;

import java.util.Arrays;


public class MnistBlock extends SequentialBlock {
    private MnistBlock() {
    }

    public static MnistBlock newMnistBlock() {
        return (MnistBlock) new MnistBlock()
            .add(Conv2d.builder()
                .setFilters(8)
                .setKernelShape(new Shape(5, 5))
                .optBias(false)
                .optPadding(new Shape(2, 2))
                .build())
            .add(Activation.reluBlock())
            .add(Pool.maxPool2dBlock(new Shape(2, 2), new Shape(2, 2)))   // 28 -> 14
            .add(Activation.reluBlock())
            .add(Pool.maxPool2dBlock(new Shape(2, 2), new Shape(2, 2)))  // 14 -> 7
            .add(Conv2d.builder()
                .setFilters(32)
                .setKernelShape(new Shape(3, 3))
                .optBias(false)
                .optPadding(new Shape(1, 1))
                .build())
            .add(Activation.reluBlock())
            .add(Blocks.batchFlattenBlock())
            .add(Linear.builder()
                .setUnits(10)
                .optBias(true)
                .build());
    }
    public static MnistBlock newMnistBlock2() {
        return (MnistBlock) new MnistBlock().add(
            new Mlp(
                Mnist.IMAGE_HEIGHT * Mnist.IMAGE_WIDTH,
                Mnist.NUM_CLASSES,
                new int[] {128, 64})
        );
    }


}
