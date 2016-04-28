package bitcoin;

import java.io.BufferedReader;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer.Builder;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import bitcoin.BitcoinFeedForward.TestStatistic;
import ch.qos.logback.classic.pattern.FileOfCallerConverter;

public class BitcoinFeedForward {

	private ExecutorService executor = Executors.newFixedThreadPool(3);

	static final double factor = 800;
	static final Random r = new Random(7894);
	static final List<Double> values = new ArrayList<>();
	static final List<Long> timestamps = new ArrayList<>();

	public static void main(String[] args) throws IOException, Exception {
		// load values
		BufferedReader is = new BufferedReader(new FileReader("courses_bpi_fixed.log"));
		String line;
		while ((line = is.readLine()) != null) {
			String[] splits = line.split(",");
			values.add(Double.valueOf(splits[1]) / factor);
			timestamps.add(Long.valueOf(splits[0]));
		}
		is.close();
		System.out.println(values.size());

		//testHyperParameters(8342, 2022, 0.006250, 3, 106, 1, 3, new GaussianDistribution(0, 0.174733));
		//System.exit(0);

		//testConfigString("historycount:6148 futurecount:3951 minPlus:0,006250 hiddenLayerCount:1 hiddenLayerWidth:51 samplesPerDataSet:1 iterations:1 weightA:0,000000 weightB:0,759735");
		//System.exit(0);

		new Thread(new Runnable() {
			public void run() {
				while (true)
					try {
						runRamdomHyperParameters();
					} catch (Exception e) {
						e.printStackTrace();
					}
			}
		}).start();
		new Thread(new Runnable() {
			public void run() {
				while (true)
					try {
						runRamdomHyperParameters();
					} catch (Exception e) {
						e.printStackTrace();
					}
			}
		}).start();

	}

	public static TestStatistic testConfigString(String config) {
		String[] bits = config.split(" ");
		int historycount = Integer.valueOf(bits[0].split(":")[1]);
		int futurecount = Integer.valueOf(bits[1].split(":")[1]);
		double minPlus = Double.valueOf(bits[2].split(":")[1].replace(",", "."));
		int hiddenLayerCount = Integer.valueOf(bits[3].split(":")[1]);
		int hiddenLayerWidth = Integer.valueOf(bits[4].split(":")[1]);
		int samplesPerDataSet = Integer.valueOf(bits[5].split(":")[1]);
		int iterations = Integer.valueOf(bits[6].split(":")[1]);
		double weightA = Double.valueOf(bits[7].split(":")[1].replace(",", "."));
		double weightB = Double.valueOf(bits[8].split(":")[1].replace(",", "."));
		return testHyperParameters(historycount, futurecount, minPlus, hiddenLayerCount, hiddenLayerWidth, samplesPerDataSet, iterations, new GaussianDistribution(weightA, weightB));
	}

	private static void runRamdomHyperParameters() throws IOException {
		int historycount = (int) (1440 * (r.nextDouble() * 7));
		int futurecount = (int) (1440 * (r.nextDouble() * 3));
		double minPlus = 5 / factor;
		int hiddenLayerCount = 1 + r.nextInt(6);
		int hiddenLayerWidth = 10 + r.nextInt(150);
		int samplesPerDataSet = 1 + r.nextInt(3);
		int iterations = 1 + r.nextInt(3);
		double weightA = 0;
		double weightB = r.nextDouble();

		GaussianDistribution weightDist = new GaussianDistribution(weightA, weightB);
		TestStatistic result = testHyperParameters(historycount, futurecount, minPlus, hiddenLayerCount, hiddenLayerWidth, samplesPerDataSet, iterations, weightDist);
		if (result.predictedPositive > 0 && result.predictedPositive < result.expectPositive) {
			double success = (double) result.predictedPositive / (result.predictedPositive + result.falsePositive);
			if (success > 0.7) {
				writeRecord(String.format("success:%f positivesExpected:%d foundPositived:%d falsePositives:%d historycount:%d futurecount:%d minPlus:%f hiddenLayerCount:%d hiddenLayerWidth:%d samplesPerDataSet:%d iterations:%d weightA:%f weightB:%f", success, result.expectPositive, result.predictedPositive, result.falsePositive, historycount, futurecount, minPlus, hiddenLayerCount, hiddenLayerWidth, samplesPerDataSet, iterations, weightA, weightB));
			}
		}
	}

	private synchronized static void writeRecord(String s) throws IOException {
		PrintWriter pw = new PrintWriter(new FileWriter(java.net.InetAddress.getLocalHost().getHostName() + "_records.txt", true));
		pw.println(s);
		pw.close();
		System.out.println(s);
	}

	private static TestStatistic testHyperParameters(int historycount, int futurecount, double minPlus, int hiddenLayerCount, int hiddenLayerWidth, int samplesPerDataSet, int iterations, GaussianDistribution weightDist) {
		NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder();
		builder.iterations(iterations);
		builder.learningRate(1e-5);
		builder.seed(123);

		// builder.regularization(true);
		// builder.useDropConnect(true);
		// builder.l1(2e-5);
		// builder.l2(2e-8);
		// builder.momentum(10);

		builder.updater(Updater.SGD);

		builder.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
		builder.biasInit(0);
		builder.miniBatch(true);
		builder.weightInit(WeightInit.DISTRIBUTION);
		builder.dist(weightDist);
		builder.activation("relu");
		// builder.gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue);

		ListBuilder listBuilder = builder.list();
		for (int i = 0; i < hiddenLayerCount; i++) {
			DenseLayer.Builder hiddenLayerBuilder = new DenseLayer.Builder();
			hiddenLayerBuilder.nIn(i == 0 ? historycount : hiddenLayerWidth);
			hiddenLayerBuilder.nOut(hiddenLayerWidth);
			listBuilder.layer(i, hiddenLayerBuilder.build());
		}

		Builder outputLayerBuilder = new OutputLayer.Builder(LossFunctions.LossFunction.SQUARED_LOSS);
		outputLayerBuilder.nIn(hiddenLayerWidth);
		outputLayerBuilder.nOut(1);
		outputLayerBuilder.activation("sigmoid");
		listBuilder.layer(hiddenLayerCount, outputLayerBuilder.build());

		listBuilder.pretrain(false);
		listBuilder.backprop(true);
		listBuilder.build();

		MultiLayerConfiguration conf = listBuilder.build();
		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();
		net.setListeners(new ScoreIterationListener(1));

		for (int epoch = 0; epoch < 2000; epoch++) {
			DataSet ds = createDataset(samplesPerDataSet, historycount, futurecount, minPlus);
			net.fit(ds);
		}

		TestStatistic testStatistic = new TestStatistic();
		testStatistic.network = net;
		for (int t = 0; t < 10000; t++) {
			testRun(net, testStatistic, historycount, futurecount, minPlus);
		}
		System.out.println("Expected positives " + testStatistic.expectPositive);
		System.out.println("Found    positives " + testStatistic.predictedPositive);
		System.out.println("False    positives " + testStatistic.falsePositive);
		System.out.println("Ratio    positives " + ((double) testStatistic.predictedPositive / (testStatistic.predictedPositive + testStatistic.falsePositive)));

		return testStatistic;
	}

	final static class TestStatistic {
		MultiLayerNetwork network;
		int expectPositive = 0;
		int falsePositive = 0;
		int predictedPositive = 0;
	}

	private static void testRun(MultiLayerNetwork net, TestStatistic testStatistic, int historycount, int futurecount, double minPlus) {
		int startindex = (int) ((values.size() - (historycount + futurecount)) * r.nextDouble());
		INDArray input = Nd4j.zeros(1, historycount);
		INDArray labels = Nd4j.zeros(1, 1);
		fillFromStartIndex(input, labels, 0, startindex, historycount, futurecount, minPlus);
		INDArray testOutput = net.output(input);

		if (labels.getDouble(0, 0) == 1)
			testStatistic.expectPositive++;

		if (labels.getDouble(0, 0) == 1 && testOutput.getDouble(0, 0) > 0.9)
			testStatistic.predictedPositive++;

		if (labels.getDouble(0, 0) == 0 && testOutput.getDouble(0, 0) > 0.9)
			testStatistic.falsePositive++;

	}

	private static DataSet createDataset(int samplesPerDataset, int historycount, int futurecount, double minPlus) {
		INDArray input = Nd4j.zeros(samplesPerDataset, historycount);
		INDArray labels = Nd4j.zeros(samplesPerDataset, 1);
		for (int sampleindex = 0; sampleindex < samplesPerDataset; sampleindex++) {
			int startindex = (int) ((values.size() - (historycount + futurecount)) * r.nextDouble());
			fillFromStartIndex(input, labels, sampleindex, startindex, historycount, futurecount, minPlus);
		}
		DataSet ds = new DataSet(input, labels);
		// System.out.println((double) positives / total);
		return ds;
	}

	private static void fillFromStartIndex(INDArray input, INDArray labels, int sampleindex, int startindex, int historycount, int futurecount, double minPlus) {

		double firstvalue = values.get(startindex);
		for (int samplePos = 0; samplePos < historycount; samplePos++) {
			input.putScalar(new int[] { sampleindex, samplePos }, values.get(startindex + samplePos) - firstvalue);
		}

		labels.putScalar(new int[] { sampleindex, 0 }, 0);
		double lastvalue = values.get(startindex + (historycount - 1));
		for (int futureindex = 0; futureindex < futurecount; futureindex++) {
			Double newValue = values.get(startindex + historycount + futureindex);
			if (newValue - lastvalue > minPlus) {

				// System.out.println(new Date(timestamps.get(startindex +
				// (historycount - 1))) + " " + lastvalue);
				// System.out.println(new Date(timestamps.get(startindex +
				// historycount + futureindex)) + " " + newValue);

				labels.putScalar(new int[] { sampleindex, 0 }, 1);
				break;
			}
		}
	}
}
