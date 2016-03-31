package lotto;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

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
 * @author Peter Großmann
 */
public class LottoRNNExample {

	public static final int HIDDEN_LAYER_WIDTH = 50;
	public static final int HIDDEN_LAYER_CONT = 2;
	public static final Random r = new Random(7894);

	public static void main(String[] args) throws NumberFormatException, IOException {

		// load values
		BufferedReader is = new BufferedReader(new FileReader("lottozahlen_archiv.csv"));
		String line;
		// die ersten beiden zeilen sind schrott
		List<String> lines = new ArrayList<>();
		// skip first 2 lines
		is.readLine();
		is.readLine();
		while ((line = is.readLine()) != null) {
			if (line.split(",").length == 10)
				lines.add(line);
		}
		is.close();
		System.out.println(lines.size());
		Collections.reverse(lines);

		INDArray input = Nd4j.zeros(1, 49, lines.size() - 1);
		INDArray labels = Nd4j.zeros(1, 49, lines.size() - 1);
		for (int i = 0; i < lines.size() - 1; i++) {
			String[] current = lines.get(i).split(",");
			String[] next = lines.get(i + 1).split(",");
			for (int j = 0; j < 6; j++) {
				input.putScalar(new int[] { 0, Integer.valueOf(current[j + 2]) - 1, i }, 1);
				labels.putScalar(new int[] { 0, Integer.valueOf(next[j + 2]) - 1, i }, 1);
			}
		}

		DataSet ds = new DataSet(input, labels);

		NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder();
		builder.iterations(10);
		builder.learningRate(0.01);
		builder.regularization(true);
		builder.l1(0.1);
		builder.l2(0.1);
		builder.useDropConnect(true);
		builder.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
		builder.seed(123);
		builder.biasInit(0);
		builder.miniBatch(false);
		builder.updater(Updater.RMSPROP);
		builder.weightInit(WeightInit.XAVIER);
		// builder.dist(new UniformDistribution(-0.1, 0.1));

		ListBuilder listBuilder = builder.list(HIDDEN_LAYER_CONT + 1);

		for (int i = 0; i < HIDDEN_LAYER_CONT; i++) {
			GravesLSTM.Builder hiddenLayerBuilder = new GravesLSTM.Builder();
			hiddenLayerBuilder.nIn(i == 0 ? 49 : HIDDEN_LAYER_WIDTH);
			hiddenLayerBuilder.nOut(HIDDEN_LAYER_WIDTH);
			hiddenLayerBuilder.activation("tanh");
			listBuilder.layer(i, hiddenLayerBuilder.build());
		}

		RnnOutputLayer.Builder outputLayerBuilder = new RnnOutputLayer.Builder(LossFunction.MCXENT);
		outputLayerBuilder.activation("softmax");
		outputLayerBuilder.nIn(HIDDEN_LAYER_WIDTH);
		outputLayerBuilder.nOut(49);
		listBuilder.layer(HIDDEN_LAYER_CONT, outputLayerBuilder.build());

		listBuilder.pretrain(false);
		listBuilder.backprop(true);
		listBuilder.build();

		MultiLayerConfiguration conf = listBuilder.build();
		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();
		net.setListeners(new ScoreIterationListener(1));

		for (int i = 0; i < 10000; i++) {
			net.fit(ds);

			net.rnnClearPreviousState();


			INDArray output = net.rnnTimeStep(ds.getFeatureMatrix());
			INDArray trimoutput = output.tensorAlongDimension(output.size(2)-1,1,0);
			
			
			List<Integer> lottozahlen = IntStream.rangeClosed(1, 49).boxed().sorted((f1,f2) -> Double.compare(trimoutput.getDouble(f1-1),trimoutput.getDouble(f2-1))).limit(6).collect(Collectors.toList());  
			
			System.out.println(lottozahlen);

		}

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

}
