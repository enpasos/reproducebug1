package com.enpasos.bugs;

import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.Mnist;
import ai.djl.engine.Engine;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.listener.SaveModelTrainingListener;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import lombok.extern.slf4j.Slf4j;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;


/**
 * An example of training an image classification (MNIST) model.
 *
 * <p>See this <a
 * href="https://github.com/deepjavalibrary/djl/blob/master/examples/docs/train_mnist_mlp.md">doc</a>
 * for information about this example.
 */
@Slf4j
public final class Main {

    private Main() {
    }

    public static void main(String[] args) throws IOException, TranslateException {
        String[] args2 = {"-e", "1", "-b", "10", "-o", "mymodel"};
        Main.runExample(args2);
    }

    public static void runExample(String[] args) throws IOException, TranslateException {
        Arguments arguments = new Arguments().parseArgs(args);
        if (arguments == null) {
            return;
        }

        RandomAccessDataset trainingSet = getDataset(Dataset.Usage.TRAIN, arguments);
        RandomAccessDataset validateSet = getDataset(Dataset.Usage.TEST, arguments);

        Block block = MnistBlock.newMnistBlock();

        try (Model model = Model.newInstance("mymodel")) {
            model.setBlock(block);

            DefaultTrainingConfig config = setupTrainingConfig(arguments);
            Shape inputShape = new Shape(1, 1, Mnist.IMAGE_HEIGHT, Mnist.IMAGE_WIDTH);

            List<DurAndMem> durations = new ArrayList<>();

            try (Trainer trainer = model.newTrainer(config)) {
                trainer.setMetrics(new Metrics());
                trainer.initialize(inputShape);

                for (int epoch = 0; epoch < 10; epoch++) {
                    log.info("Training epoch = {}", epoch);
                    DurAndMem duration = new DurAndMem();
                    duration.on();

                    EasyTrain.fit(trainer, 1, trainingSet, validateSet);

                    duration.off();
                    durations.add(duration);
                    System.out.println("epoch;duration[s];gpuMem[MiB]");
                    IntStream.range(0, durations.size()).forEach(i -> System.out.println(i + ";" + durations.get(i).getDur() / 1000 + ";" + durations.get(i).getMem()/1024/1024));
                }
            }
        }
    }

    private static DefaultTrainingConfig setupTrainingConfig(Arguments arguments) {
        String outputDir = arguments.getOutputDir();
        SaveModelTrainingListener listener = new SaveModelTrainingListener(outputDir);
        return new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
            .optDevices(Engine.getInstance().getDevices(arguments.getMaxGpus()))
            .addTrainingListeners(TrainingListener.Defaults.logging(outputDir))
            .addTrainingListeners(listener);
    }

    private static RandomAccessDataset getDataset(Dataset.Usage usage, Arguments arguments)
        throws IOException {
        Mnist mnist =
            Mnist.builder()
                .optUsage(usage)
                .setSampling(arguments.getBatchSize(), true)
                .optLimit(arguments.getLimit())
                .build();
        mnist.prepare(new ProgressBar());
        return mnist;
    }
}
