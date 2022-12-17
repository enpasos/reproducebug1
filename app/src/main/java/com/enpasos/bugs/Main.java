package com.enpasos.bugs;

import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.Mnist;
import ai.djl.engine.Engine;
import ai.djl.metric.Metrics;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.BaseNDManager;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.pytorch.engine.PtNDManager;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.SaveModelTrainingListener;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import lombok.extern.slf4j.Slf4j;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;


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


        boolean isGarbageCollectionOn = false;

        if(args.length > 0 && args[0].equals("gc")) {
            isGarbageCollectionOn = true;
        }

        String[] args2 = {"-e", "10", "-b", "10", "-o", "mymodel"};


        Arguments arguments = new Arguments().parseArgs(args2);
        if (arguments == null) {
            return;
        }

        RandomAccessDataset trainingSet = getDataset(Dataset.Usage.TRAIN, arguments);
        RandomAccessDataset validateSet = getDataset(Dataset.Usage.TEST, arguments);


        Map<String, List<Image>> data = getData("./app/data/");

        Block block = MnistBlock.newMnistBlock();

        try (Model model = Model.newInstance("mymodel", isGarbageCollectionOn)) {
            model.setBlock(block);

            DefaultTrainingConfig config = setupTrainingConfig(arguments);
            Shape inputShape = new Shape(1, 1, Mnist.IMAGE_HEIGHT, Mnist.IMAGE_WIDTH);

            List<DurAndMem> durations = new ArrayList<>();

            try (Trainer trainer = model.newTrainer(config)) {
                trainer.setMetrics(new Metrics());
                trainer.initialize(inputShape);
                for (int epoch = 0; epoch < arguments.epoch; epoch++) {

                    // training
                    log.info("Training epoch = {}", epoch);
                    DurAndMem duration = new DurAndMem();
                    duration.on();

                    EasyTrain.fit(trainer, 1, trainingSet, validateSet);

                    duration.off();
                    durations.add(duration);
                    System.out.println("epoch;duration[ms];gpuMem[MiB]");
                    IntStream.range(0, durations.size()).forEach(i -> System.out.println(i + ";" + durations.get(i).getDur() + ";" + durations.get(i).getMem() / 1024 / 1024));


                    // inference
                    try (var predictor = model.newPredictor(getImageClassificationsTranslator())) {

                        int[] errorsTotal = {0, 0};
                        data.forEach((label, images) -> images.forEach(image -> {
                            try {
                                var classifications = predictor.predict(image);
                                if (!classifications.best().getClassName().equals(label)) {
                                    errorsTotal[0]++;
                                }
                                errorsTotal[1]++;
                            } catch (Exception e) {
                                e.printStackTrace();
                                throw new RuntimeException(e);
                            }
                        }));


                        log.info("{} wrong classified images in {} non trained testimages", errorsTotal[0], errorsTotal[1]);
                    }
                    ((BaseNDManager) model.getNDManager().getParentManager()).debugDump(0);

                }
            }
        }
    }

    private static Map<String, List<Image>> getData(String dataPath) {
        Map<String, List<Image>> data = new TreeMap<>();
        try (Stream<Path> stream = Files.list(Paths.get(dataPath))) {
            stream.filter(Files::isDirectory)
                .map(Path::getFileName)
                .forEach(dirname -> {
                    List<Image> images = new ArrayList<>();
                    data.put(dirname.toString(), images);
                    try (Stream<Path> stream2 = Files.list(Paths.get(dataPath + dirname + "/"))) {
                        stream2
                            .filter(file -> !Files.isDirectory(file))
                            .forEach(path -> {
                                try {
                                    images.add(ImageFactory.getInstance().fromFile(path));
                                } catch (Exception e) {
                                    e.printStackTrace();
                                    throw new RuntimeException(e);
                                }
                            });
                    } catch (Exception e) {
                        e.printStackTrace();
                        throw new RuntimeException(e);
                    }
                });
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException(e);
        }
        return data;
    }

    private static Translator<Image, Classifications> getImageClassificationsTranslator() {
        return new Translator<>() {

            @Override
            public NDList processInput(TranslatorContext ctx, Image input) {
                // Convert Image to NDArray
                NDArray array = input.toNDArray(ctx.getNDManager(), Image.Flag.GRAYSCALE);
                return new NDList(NDImageUtils.toTensor(array));
            }

            @Override
            public Classifications processOutput(TranslatorContext ctx, NDList list) {
                // Create a Classifications with the output probabilities
                NDArray probabilities = list.singletonOrThrow().softmax(0);
                List<String> classNames = IntStream.range(0, 10).mapToObj(String::valueOf).collect(Collectors.toList());
                return new Classifications(classNames, probabilities);
            }

            @Override
            public Batchifier getBatchifier() {
                // The Batchifier describes how to combine a batch together
                // Stacking, the most common batchifier, takes N [X1, X2, ...] arrays to a single [N, X1, X2, ...] array
                return Batchifier.STACK;
            }
        };
    }


    private static DefaultTrainingConfig setupTrainingConfig(Arguments arguments) {
        String outputDir = arguments.getOutputDir();
        SaveModelTrainingListener listener = new SaveModelTrainingListener(outputDir);
        return new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
            .addEvaluator(new Accuracy())
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
