package bitcoin;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

/**
 * @author Peter Gro�mann
 */
public class SaveLoad {

    public static final int HIDDEN_LAYER_WIDTH = 200;
    public static final int HIDDEN_LAYER_CONT = 2;
    public static final Random r = new Random(78945);

    public static void main(String[] args) throws NumberFormatException, IOException, ClassNotFoundException {

        NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder();
        builder.iterations(1);
        builder.learningRate(0.01);
        builder.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
        builder.seed(123);

        builder.biasInit(0);
        builder.miniBatch(true);
        builder.updater(Updater.RMSPROP);

        builder.weightInit(WeightInit.DISTRIBUTION);
        builder.dist(new UniformDistribution(-1, 1));

        ListBuilder listBuilder = builder.list(HIDDEN_LAYER_CONT + 1);

        for (int i = 0; i < HIDDEN_LAYER_CONT; i++) {
            GravesLSTM.Builder hiddenLayerBuilder = new GravesLSTM.Builder();
            hiddenLayerBuilder.nIn(i == 0 ? 3 : HIDDEN_LAYER_WIDTH);
            hiddenLayerBuilder.nOut(HIDDEN_LAYER_WIDTH);
            hiddenLayerBuilder.activation("tanh");
            listBuilder.layer(i, hiddenLayerBuilder.build());
        }

        RnnOutputLayer.Builder outputLayerBuilder = new RnnOutputLayer.Builder(LossFunction.MCXENT);
        outputLayerBuilder.activation("softmax");
        outputLayerBuilder.nIn(HIDDEN_LAYER_WIDTH);
        outputLayerBuilder.nOut(3);
        listBuilder.layer(HIDDEN_LAYER_CONT, outputLayerBuilder.build());

        listBuilder.pretrain(false);
        listBuilder.backprop(true);
        listBuilder.build();

        MultiLayerConfiguration conf = listBuilder.build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        INDArray input = Nd4j.zeros(10, 3, 144);
        INDArray labels = Nd4j.zeros(10, 3, 144);

        net.fit(new DataSet(input, labels));
        net.fit(new DataSet(input, labels));
        save(net);

        net = load();
        net.fit(new DataSet(input, labels));
        System.out.println("works");
    }

    private static void save(MultiLayerNetwork net) throws IOException {
        // Write the network parameters:
        try (DataOutputStream dos = new DataOutputStream(Files.newOutputStream(Paths.get("coefficients.bin")))) {
            Nd4j.write(net.params(), dos);
        }

        // Write the network configuration:
        FileUtils.write(new File("conf.json"), net.getLayerWiseConfigurations().toJson());

        // Save the updater:
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream("updater.bin"))) {
            oos.writeObject(net.getUpdater());
        }
    }

    private static MultiLayerNetwork load() throws FileNotFoundException, IOException, ClassNotFoundException {
        // Load network configuration from disk:
        MultiLayerConfiguration confFromJson = MultiLayerConfiguration.fromJson(FileUtils.readFileToString(new File("conf.json")));

        // Load parameters from disk:
        INDArray newParams;
        try (DataInputStream dis = new DataInputStream(new FileInputStream("coefficients.bin"))) {
            newParams = Nd4j.read(dis);
        }

        // Create a MultiLayerNetwork from the saved configuration and
        // parameters
        MultiLayerNetwork savedNetwork = new MultiLayerNetwork(confFromJson);
        savedNetwork.init();
        // must be set before setUpdater()
        savedNetwork.setParameters(newParams);

        // Load the updater:
        org.deeplearning4j.nn.api.Updater updater;
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream("updater.bin"))) {
            updater = (org.deeplearning4j.nn.api.Updater) ois.readObject();
        }

        // Set the updater in the network
        savedNetwork.setUpdater(updater);

        return savedNetwork;
    }

}