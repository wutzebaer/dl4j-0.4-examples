package bitcoin;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Random;
import java.util.concurrent.LinkedBlockingQueue;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
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

public class BitcoinFeedForward {

	static final double factor = 800;
	static final Random r = new Random(7894);
	static final int historycount = 1440 / 30 * 7;
	static final int futurecount = 1440 / 30 * 2;
	static final double minPlus = 4.5 / factor;
	static final List<Double> values = new ArrayList<>();
	static final List<Long> timestamps = new ArrayList<>();
	static final LinkedBlockingQueue<DataSet> asyncInserts = new LinkedBlockingQueue<>(2);
	static final int hiddenLayerCount = 100;
	static final int hiddenLayerWidth = 50;

	static final int samplesPerDataSet = 1000;

	public static void main(String[] args) throws IOException, Exception {
		// load values
		BufferedReader is = new BufferedReader(new FileReader("courses_bpi_fixed_merged.log"));
		String line;
		while ((line = is.readLine()) != null) {
			String[] splits = line.split(",");
			values.add(Double.valueOf(splits[1]) / factor);
			timestamps.add(Long.valueOf(splits[0]));
		}
		is.close();
		System.out.println(values.size());

		NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder();
		builder.iterations(20);
		builder.learningRate(1e-2);
		builder.seed(123);

		// builder.setDropOut(0.5);

		//builder.setUseRegularization(true);
		//builder.l1(1);
		//builder.l2(1);

		builder.momentum(0.9);
		builder.updater(Updater.NESTEROVS);

		builder.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
		builder.biasInit(1);
		builder.miniBatch(true);
		builder.weightInit(WeightInit.XAVIER);
		builder.activation("sigmoid");
		//builder.gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue);

		ListBuilder listBuilder = builder.list();
		for (int i = 0; i < hiddenLayerCount; i++) {
			DenseLayer.Builder hiddenLayerBuilder = new DenseLayer.Builder();
			hiddenLayerBuilder.nIn(i == 0 ? historycount : hiddenLayerWidth);
			hiddenLayerBuilder.nOut(hiddenLayerWidth);
			listBuilder.layer(i, hiddenLayerBuilder.build());
		}

		Builder outputLayerBuilder = new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT);
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

		spawnSampleThread();
		spawnSampleThread();

		for (int epoch = 0; epoch < 100; epoch++) {
			DataSet ds = asyncInserts.take();
			net.fit(ds);
		}

		while (true) {
			testRun(net);
		}
	}

	private static void spawnSampleThread() {
		new Thread(new Runnable() {
			public void run() {
				try {
					while (true)
						asyncInserts.put(createDataset(samplesPerDataSet));
				} catch (InterruptedException e) {
					throw new RuntimeException(e);
				}
			}
		}).start();
	}

	private static void testRun(MultiLayerNetwork net) {
		int startindex = (int) ((values.size() - (historycount + futurecount)) * r.nextDouble());
		INDArray input = Nd4j.zeros(1, historycount);
		INDArray labels = Nd4j.zeros(1, 1);
		fillFromStartIndex(input, labels, 0, startindex);
		INDArray testOutput = net.output(input);
		
		System.out.println(new Date(timestamps.get(startindex)));
		System.out.println("sollte: " + labels.getDouble(0, 0));
		System.out.println("ist   : " + testOutput.getDouble(0, 0));
	}

	private static DataSet createDataset(int samplesPerDataset) {
		INDArray input = Nd4j.zeros(samplesPerDataset, historycount);
		INDArray labels = Nd4j.zeros(samplesPerDataset, 1);
		int positives = 0;
		int total = 0;
		for (int sampleindex = 0; sampleindex < samplesPerDataset; sampleindex++) {
			int startindex = (int) ((values.size() - (historycount + futurecount)) * r.nextDouble());
			fillFromStartIndex(input, labels, sampleindex, startindex);
			if (labels.getDouble(sampleindex, 0) == 1) {
				positives++;
			}
			total++;
		}
		DataSet ds = new DataSet(input, labels);
		System.out.println((double) positives / total);
		return ds;
	}

	private static void fillFromStartIndex(INDArray input, INDArray labels, int sampleindex, int startindex) {
		
		double firstvalue = values.get(startindex);
		for (int samplePos = 0; samplePos < historycount; samplePos++) {
			input.putScalar(new int[] { sampleindex, samplePos }, values.get(startindex + samplePos) - firstvalue);
		}
		
		labels.putScalar(new int[] { sampleindex, 0 }, 0);
		double lastvalue = values.get(startindex + (historycount - 1));
		for (int futureindex = 0; futureindex < futurecount; futureindex++) {
			Double newValue = values.get(startindex + historycount + futureindex);
			if (newValue - lastvalue > minPlus) {

				// System.out.println(new Date(timestamps.get(startindex + (historycount - 1))) + " " + lastvalue);
				// System.out.println(new Date(timestamps.get(startindex + historycount + futureindex)) + " " + newValue);

				labels.putScalar(new int[] { sampleindex, 0 }, 1);
				break;
			}
		}
	}
}
