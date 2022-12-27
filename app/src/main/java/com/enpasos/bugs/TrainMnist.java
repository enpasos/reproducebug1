/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package com.enpasos.bugs;

import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.Mnist;
import ai.djl.basicmodelzoo.basic.Mlp;
import ai.djl.engine.Engine;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.gc.SwitchGarbageCollection;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.pytorch.jni.JniUtils;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.SaveModelTrainingListener;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import ai.djl.util.cuda.CudaUtils;
import lombok.extern.slf4j.Slf4j;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

import static ai.djl.pytorch.engine.PtNDManager.debugDumpFromSystemManager;
import static com.enpasos.bugs.DurAndMem.calculateMem;

/**
 * An example of training an image classification (MNIST) model.
 *
 * <p>See this <a
 * href="https://github.com/deepjavalibrary/djl/blob/master/examples/docs/train_mnist_mlp.md">doc</a>
 * for information about this example.
 */
@Slf4j
public final class TrainMnist {

    private TrainMnist() {}

    public static void main(String[] args) throws IOException, TranslateException, InterruptedException {

        if(args.length > 0 && args[0].equals("gc")) {
            SwitchGarbageCollection.on();
        }

        String[] args2 = {"-e", "10", "-b", "10", "-o", "mymodel"};


        Arguments arguments = new Arguments().parseArgs(args2);
        if (arguments == null) {
            return;
        }

      //  for (int W = 0; W < 2; W++) {


            // Construct neural network
            Block block =
                new Mlp(
                    Mnist.IMAGE_HEIGHT * Mnist.IMAGE_WIDTH,
                    Mnist.NUM_CLASSES,
                    new int[]{128, 64});

            try (Model model = Model.newInstance("mlp")) {
                model.setBlock(block);

                // get training and validation dataset
                RandomAccessDataset trainingSet = getDataset(Dataset.Usage.TRAIN, arguments, model.getNDManager());
                RandomAccessDataset validateSet = getDataset(Dataset.Usage.TEST, arguments, model.getNDManager());

                // setup training configuration
                DefaultTrainingConfig config = setupTrainingConfig(arguments);

                try (Trainer trainer = model.newTrainer(config)) {
                    trainer.setMetrics(new Metrics());

                    /*
                     * MNIST is 28x28 grayscale image and pre processed into 28 * 28 NDArray.
                     * 1st axis is batch axis, we can use 1 for initialization.
                     */
                    Shape inputShape = new Shape(1, Mnist.IMAGE_HEIGHT * Mnist.IMAGE_WIDTH);

                    // initialize trainer with proper input shape
                    trainer.initialize(inputShape);

                    List<DurAndMem> durations = new ArrayList<>();

                    for (int epoch = 0; epoch < arguments.epoch; epoch++) {

                        // training
                        log.info("Training epoch = {}", epoch);
                        DurAndMem duration = new DurAndMem();
                        duration.on();
                        // We iterate through the dataset once during each epoch
                        for (Batch batch : trainer.iterateDataset(trainingSet)) {

                            EasyTrain.trainBatch(trainer, batch);

                            trainer.step();
                            batch.close();
                        }

                        EasyTrain.evaluateDataset(trainer, validateSet);

                        // reset training and validation evaluators at end of epoch
                        trainer.notifyListeners(listener -> listener.onEpoch(trainer));

                        duration.off();
                        durations.add(duration);
                        System.out.println("epoch;duration[ms];gpuMem[MiB]");
                        IntStream.range(0, durations.size()).forEach(i -> System.out.println(i + ";" + durations.get(i).getDur() + ";" + durations.get(i).getMem() / 1024 / 1024));

                        System.gc(); // just for testing - do not use in production
                        TimeUnit.SECONDS.sleep(1);
                        debugDumpFromSystemManager(false);

                        log.info("gpuMem[MiB]: {}", calculateMem() / 1024 / 1024);

                    }

                }

                log.info("closed trainer");
                System.gc(); // just for testing - do not use in production
                TimeUnit.SECONDS.sleep(1);
                debugDumpFromSystemManager(false);
                log.info("gpuMem[MiB]: {}", calculateMem() / 1024 / 1024);
            }
            log.info("closed model");
            System.gc(); // just for testing - do not use in production
            TimeUnit.SECONDS.sleep(1);
            debugDumpFromSystemManager(false);
            //for (int i = 0; i < 10000; i++) {
            log.info("gpuMem[MiB] after {}s: {}", 0, calculateMem() / 1024 / 1024);
            TimeUnit.SECONDS.sleep(1);

            log.info("gpuMem[MiB] after {}s and cudaDeviceReset(): {}", 1, calculateMem() / 1024 / 1024);
            //}
        }
    //}

    private static DefaultTrainingConfig setupTrainingConfig(Arguments arguments) {
        String outputDir = arguments.getOutputDir();
        SaveModelTrainingListener listener = new SaveModelTrainingListener(outputDir);
        listener.setSaveModelCallback(
            trainer -> {
                TrainingResult result = trainer.getTrainingResult();
                Model model = trainer.getModel();
                float accuracy = result.getValidateEvaluation("Accuracy");
                model.setProperty("Accuracy", String.format("%.5f", accuracy));
                model.setProperty("Loss", String.format("%.5f", result.getValidateLoss()));
            });
        return new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
            .addEvaluator(new Accuracy())
            .optDevices(Engine.getInstance().getDevices(arguments.getMaxGpus()))
            .addTrainingListeners(TrainingListener.Defaults.logging(outputDir))
            .addTrainingListeners(listener);
    }

    private static RandomAccessDataset getDataset(Dataset.Usage usage, Arguments arguments, NDManager manager)
        throws IOException {
        Mnist mnist =
            Mnist.builder()
                .optUsage(usage)
                .optManager(manager)
                .setSampling(arguments.getBatchSize(), true)
                .optLimit(arguments.getLimit())
                .build();
        mnist.prepare(new ProgressBar());
        return mnist;
    }
}
