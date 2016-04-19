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
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.apache.commons.io.FileUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.storage.StorageLevel;
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
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.json.JSONException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

/**
 * @author Peter Gro�mann
 */
public class BitcoinRNNExampleHilo {

	public static final int HIDDEN_LAYER_WIDTH = 80;
	public static final int HIDDEN_LAYER_CONT = 2;
	public static final Random r = new Random(78945);
	public static final int sampleLangth = 1440 / 30 * 1; // 10080=7 tage 1440=1
															// tag
	public static final int samplesPerDataset = 1000;

	public static void main(String[] args) throws NumberFormatException, IOException, JSONException {

		System.out.println(System.getProperty("java.library.path"));
		
		// BitcoinChartLOader.main(args);

		NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder();
		builder.iterations(100);
		builder.learningRate(0.01);
		builder.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
		builder.seed(123);
		// builder.regularization(true);
		// builder.l1(0.9);
		// builder.l2(0.1);
		// builder.momentum(0.8);

		builder.biasInit(0);
		builder.miniBatch(true);

		// builder.weightInit(WeightInit.XAVIER);
		builder.weightInit(WeightInit.DISTRIBUTION);
		builder.dist(new UniformDistribution(-1, 1));

		ListBuilder listBuilder = builder.list(HIDDEN_LAYER_CONT + 1);

		for (int i = 0; i < HIDDEN_LAYER_CONT; i++) {
			GravesLSTM.Builder hiddenLayerBuilder = new GravesLSTM.Builder();
			hiddenLayerBuilder.nIn(i == 0 ? 3 : HIDDEN_LAYER_WIDTH);
			hiddenLayerBuilder.nOut(HIDDEN_LAYER_WIDTH);
			hiddenLayerBuilder.activation("tanh");
			hiddenLayerBuilder.updater(Updater.RMSPROP);
			listBuilder.layer(i, hiddenLayerBuilder.build());
		}

		RnnOutputLayer.Builder outputLayerBuilder = new RnnOutputLayer.Builder(LossFunction.MCXENT);
		outputLayerBuilder.activation("softmax");
		outputLayerBuilder.nIn(HIDDEN_LAYER_WIDTH);
		outputLayerBuilder.nOut(3);
		outputLayerBuilder.updater(Updater.RMSPROP);
		listBuilder.layer(HIDDEN_LAYER_CONT, outputLayerBuilder.build());

		listBuilder.pretrain(false);
		listBuilder.backprop(true);
		listBuilder.build();

		MultiLayerConfiguration conf = listBuilder.build();
		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();
		net.setListeners(new ScoreIterationListener(10));
		net.setUpdater(null);

		try {
			net = load();
		} catch (Exception e) {
			System.out.println("net not loaded " + e.getMessage());
		}

		List<Integer> values = loadValues();

		final boolean SPARK = false;
		SparkDl4jMultiLayer sparkNetwork;
		JavaSparkContext sc;
		int nCores = 16;

		if (SPARK) {
			// spark config
			SparkConf sparkConf = new SparkConf();
			sparkConf.setMaster("local[" + nCores + "]");
			sparkConf.setAppName("LSTM_Char");
			sparkConf.set(SparkDl4jMultiLayer.AVERAGE_EACH_ITERATION, String.valueOf(true));
			sc = new JavaSparkContext(sparkConf);
			sparkNetwork = new SparkDl4jMultiLayer(sc, net);
		}

		for (int i = 0; i < 10000; i++) {

			if (SPARK) {
				// spark dataset
				List<DataSet> list = new ArrayList<>();
				for (int j = 0; j < nCores; j++) {
					list.add(createDataset(values, samplesPerDataset, sampleLangth));
				}
				JavaRDD<DataSet> datasets = sc.parallelize(list);
				datasets.persist(StorageLevel.MEMORY_ONLY());
				net = sparkNetwork.fitDataSet(datasets);
			} else {
				// normal
				DataSet learnDs = createDataset(values, samplesPerDataset, sampleLangth);
				net.fit(learnDs);
			}

			save(net);

			for (int t = 0; t < 10; t++)
				testNetwork(values, net, sampleLangth, sampleLangth, true);

			for (int t = 0; t < 10; t++)
				testNetwork(values, net, sampleLangth, sampleLangth / 2, true);

			for (int t = 0; t < 10; t++)
				testNetwork(values, net, sampleLangth / 2, sampleLangth / 2, true);

		}

	}

	protected static List<Integer> loadValues() throws FileNotFoundException, IOException {
		List<Integer> values = new ArrayList<>();
		// load values
		BufferedReader is = new BufferedReader(new FileReader("courses_bpi_fixed_merged_hilo.log"));
		String line;
		while ((line = is.readLine()) != null) {
			String[] splits = line.split(",");
			values.add(Integer.valueOf(splits[1]));
		}
		is.close();
		System.out.println(values.size());
		return values;
	}

	protected static boolean testNetwork(List<Integer> values, MultiLayerNetwork net, int samplesToInit, int samplesToPredict, boolean print) {
		net.rnnClearPreviousState();
		DataSet learnDs = createDataset(values, 1, samplesToInit + samplesToPredict);
		
		INDArray lastOutput = Nd4j.zeros(3);
		
		for (int i = 0; i < samplesToInit; i++) {
			INDArray testInput = learnDs.getFeatureMatrix().tensorAlongDimension(i, 1, 0);
			lastOutput = net.rnnTimeStep(testInput);
		}
		
		// expected balance for samplesToPredict
		int targetTotal = 0;
		// predicted balance for samplesToPredict
		int predictedTotal = 0;
		
		for (int i = samplesToInit; i < samplesToInit + samplesToPredict; i++) {
			// add to expected balance
			int target = sampleFromDistribution(learnDs.getLabels().tensorAlongDimension(i, 1, 0)) - 1;
			targetTotal += target;
			
			// add to predived balance
			int predicted = sampleFromDistribution(lastOutput) - 1;
			predictedTotal += predicted;
			
			// predict next
			lastOutput = net.rnnTimeStep(lastOutput);
		}
		
		if(print)
			System.out.println(samplesToInit + "/" + samplesToPredict + " exprected " + targetTotal + " predicted " + predictedTotal);
		
		if(targetTotal > 0 && predictedTotal > 0) return true;
		else if(targetTotal < 0 && predictedTotal < 0) return true;
		else if(targetTotal == 0 && predictedTotal == 0) return true;
		else return false;
	}
	
	protected static int testNetworkError(List<Integer> values, MultiLayerNetwork net, int samplesToInit, int samplesToPredict, boolean print) {
		net.rnnClearPreviousState();
		DataSet learnDs = createDataset(values, 1, samplesToInit + samplesToPredict);

		INDArray lastOutput = Nd4j.zeros(3);

		for (int i = 0; i < samplesToInit; i++) {
			INDArray testInput = learnDs.getFeatureMatrix().tensorAlongDimension(i, 1, 0);
			lastOutput = net.rnnTimeStep(testInput);
		}

		// expected balance for samplesToPredict
		int targetTotal = 0;
		// predicted balance for samplesToPredict
		int predictedTotal = 0;

		for (int i = samplesToInit; i < samplesToInit + samplesToPredict; i++) {
			// add to expected balance
			int target = sampleFromDistribution(learnDs.getLabels().tensorAlongDimension(i, 1, 0)) - 1;
			targetTotal += target;

			// add to predived balance
			int predicted = sampleFromDistribution(lastOutput) - 1;
			predictedTotal += predicted;

			// predict next
			lastOutput = net.rnnTimeStep(lastOutput);
		}

		if(print)
			System.out.println(samplesToInit + "/" + samplesToPredict + " exprected " + targetTotal + " predicted " + predictedTotal);

		return targetTotal - predictedTotal;
	}

	private static DataSet createDataset(List<Integer> values, int samplesPerDataset, int sampleLangth) {
		INDArray input = Nd4j.zeros(samplesPerDataset, 3, sampleLangth);
		INDArray labels = Nd4j.zeros(samplesPerDataset, 3, sampleLangth);
		for (int sampleindex = 0; sampleindex < samplesPerDataset; sampleindex++) {
			int startindex = (int) ((values.size() - 1 - sampleLangth) * r.nextDouble());
			for (int samplePos = 0; samplePos < sampleLangth; samplePos++) {
				input.putScalar(new int[] { sampleindex, values.get(startindex + samplePos) + 1, samplePos }, 1);
				labels.putScalar(new int[] { sampleindex, values.get(startindex + samplePos + 1) + 1, samplePos }, 1);
			}
		}
		DataSet ds = new DataSet(input, labels);
		return ds;
	}

	private static int sampleFromDistribution(INDArray output) {
		double d = r.nextDouble();
		double sum = 0.0;
		for (int i = 0; i < output.size(1); i++) {
			sum += output.getDouble(i);
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

		System.out.println("Net saved");
	}

	protected static MultiLayerNetwork load() throws FileNotFoundException, IOException, ClassNotFoundException {
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
		savedNetwork.setListeners(new ScoreIterationListener(10));
		savedNetwork.setParameters(newParams);

		// Load the updater:
		org.deeplearning4j.nn.api.Updater updater;
		try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream("updater.bin"))) {
			updater = (org.deeplearning4j.nn.api.Updater) ois.readObject();
		}

		// Set the updater in the network
		savedNetwork.setUpdater(updater);

		System.out.println("Net loaded");

		return savedNetwork;
	}

}
