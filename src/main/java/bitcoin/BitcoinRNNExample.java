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
public class BitcoinRNNExample {

	public static final int HIDDEN_LAYER_WIDTH = 1;
	public static final int HIDDEN_LAYER_CONT = 1;
	public static final Random r = new Random(7894);
	public static final double factor = 1;
	public static final List<Double> values = new ArrayList<>();
	public static final int sampleLangth = 48; // 10080=7 tage 1440=1 tag
	public static final int samplesPerDataset = 1;

	public static void main(String[] args) throws NumberFormatException, IOException {

		NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder();
		builder.iterations(10000);
		builder.learningRate(0.001);
		builder.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
		builder.seed(123);
		builder.regularization(true);
		// builder.l1(0.9);
		builder.l2(0.1);
		// builder.momentum(0.9);
		
		builder.biasInit(0);
		builder.miniBatch(false);
		builder.updater(Updater.RMSPROP);
		
		builder.weightInit(WeightInit.DISTRIBUTION);
		builder.dist(new UniformDistribution(-1, 1));

		ListBuilder listBuilder = builder.list(HIDDEN_LAYER_CONT + 1);

		for (int i = 0; i < HIDDEN_LAYER_CONT; i++) {
			GravesLSTM.Builder hiddenLayerBuilder = new GravesLSTM.Builder();
			hiddenLayerBuilder.nIn(i == 0 ? 1 : HIDDEN_LAYER_WIDTH);
			hiddenLayerBuilder.nOut(HIDDEN_LAYER_WIDTH);
			hiddenLayerBuilder.activation("sigmoid");
			
			listBuilder.layer(i, hiddenLayerBuilder.build());
		}

		RnnOutputLayer.Builder outputLayerBuilder = new RnnOutputLayer.Builder(LossFunction.RMSE_XENT);
		outputLayerBuilder.activation("identity");
		outputLayerBuilder.nIn(HIDDEN_LAYER_WIDTH);
		outputLayerBuilder.nOut(1);
		listBuilder.layer(HIDDEN_LAYER_CONT, outputLayerBuilder.build());

		listBuilder.pretrain(false);
		listBuilder.backprop(true);
		listBuilder.build();

		MultiLayerConfiguration conf = listBuilder.build();
		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();
		net.setListeners(new ScoreIterationListener(1));

		try {
			// net = load();
			System.out.println("net loaded");
		} catch (Exception e) {
			System.out.println("net not loaded " + e.getMessage());
		}

		// load values
		BufferedReader is = new BufferedReader(new FileReader("courses_bpi_fixed_merged.log"));
		String line;
		while ((line = is.readLine()) != null) {
			String[] splits = line.split(",");
			values.add(Double.valueOf(splits[1]));
		}
		is.close();
		System.out.println(values.size());

		for (int i = 0; i < 1; i++) {

			DataSet learnDs = createDataset(samplesPerDataset, sampleLangth);
			net.fit(learnDs);

			save(net);

			net.rnnClearPreviousState();

			String expect = "";
			String calced = "";

			DecimalFormat df = new DecimalFormat("0000.00");

			int startindex = (int) ((values.size() - 1 - sampleLangth) * r.nextDouble());
			System.out.println(df.format(values.get(startindex)));
			for (int testIndex = startindex; testIndex < startindex + sampleLangth; testIndex++) {
				expect += df.format(values.get(testIndex + 1)) + " ";

				INDArray testInit = Nd4j.zeros(1);
				testInit.putScalar(0, values.get(testIndex) * factor);
				INDArray output = net.output(testInit);
				calced += df.format(output.getDouble(0) / factor) + " ";
			}

			System.out.println("expect " + expect);
			System.out.println("calced " + calced);

			/*
			 * net.rnnClearPreviousState();
			 * 
			 * INDArray testInit = Nd4j.zeros(LEARNSTRING_CHARS_LIST.size());
			 * testInit.putScalar(LEARNSTRING_CHARS_LIST.indexOf(LEARNSTRING[0])
			 * , 1);
			 * 
			 * INDArray output = net.rnnTimeStep(testInit);
			 * 
			 * for (int j = 0; j < LEARNSTRING.length; j++) {
			 * 
			 * // get chosen character of last output double[]
			 * outputProbDistribution = new double[LEARNSTRING_CHARS.size()];
			 * for (int k = 0; k < outputProbDistribution.length; k++) {
			 * outputProbDistribution[k] = output.getDouble(k); } int
			 * sampledCharacterIdx =
			 * sampleFromDistribution(outputProbDistribution);
			 * System.out.print(LEARNSTRING_CHARS_LIST.get(sampledCharacterIdx))
			 * ;
			 * 
			 * // generate next input INDArray nextInput =
			 * Nd4j.zeros(LEARNSTRING_CHARS_LIST.size());
			 * nextInput.putScalar(sampledCharacterIdx, 1); output =
			 * net.rnnTimeStep(nextInput);
			 * 
			 * } System.out.print("\n");
			 */
		}

	}

	private static DataSet createDataset(int samplesPerDataset, int sampleLangth) {
		INDArray input = Nd4j.zeros(samplesPerDataset, 1, sampleLangth);
		INDArray labels = Nd4j.zeros(samplesPerDataset, 1, sampleLangth);
		for (int sampleindex = 0; sampleindex < samplesPerDataset; sampleindex++) {
			int startindex = (int) ((values.size() - 1 - sampleLangth) * r.nextDouble());
			for (int samplePos = 0; samplePos < sampleLangth; samplePos++) {
				input.putScalar(new int[] { sampleindex, 0, samplePos }, values.get(startindex + samplePos) * factor);
				labels.putScalar(new int[] { sampleindex, 0, samplePos }, values.get(startindex + samplePos + 1) * factor);
			}
		}
		DataSet ds = new DataSet(input, labels);
		return ds;
	}

	private static int sampleFromDistribution(double[] distribution) {
		double d = r.nextDouble();
		double sum = 0.0;
		for (int i = 0; i < distribution.length; i++) {
			sum += distribution[i];
			if (d <= sum)
				return i;
		}
		// Should never happen if distribution is a valid probability
		// distribution
		throw new IllegalArgumentException("Distribution is invalid? d=" + d + ", sum=" + sum);
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
		savedNetwork.setListeners(new ScoreIterationListener(1));
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
