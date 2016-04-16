package bitcoin;

import java.util.List;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

public class BitcoinRNNExampleHiloUse {
	public static void main(String[] args) throws Exception {
		//BitcoinChartLOader.main(args);
		List<Integer> values = BitcoinRNNExampleHilo.loadValues();
		MultiLayerNetwork net = BitcoinRNNExampleHilo.load();

		int testCount = 10000;
		int rightCount = 0;

		for (int t = 0; t < testCount; t++) {
			boolean right = BitcoinRNNExampleHilo.testNetwork(values, net, BitcoinRNNExampleHilo.sampleLangth, BitcoinRNNExampleHilo.sampleLangth, false);
			if(right) rightCount++;
		}
		System.out.println("right " + ((double)rightCount/testCount));

		rightCount = 0;
		for (int t = 0; t < testCount; t++) {
			boolean right = BitcoinRNNExampleHilo.testNetwork(values, net, BitcoinRNNExampleHilo.sampleLangth, BitcoinRNNExampleHilo.sampleLangth / 2, false);
			if(right) rightCount++;
		}
		System.out.println("right " + ((double)rightCount/testCount));

		rightCount = 0;
		for (int t = 0; t < testCount; t++) {
			boolean right = BitcoinRNNExampleHilo.testNetwork(values, net, BitcoinRNNExampleHilo.sampleLangth / 2, BitcoinRNNExampleHilo.sampleLangth / 2, false);
			if(right) rightCount++;
		}
		System.out.println("right " + ((double)rightCount/testCount));
	}
}
