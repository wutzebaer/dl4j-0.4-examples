package bitcoin;

import java.util.List;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

public class BitcoinRNNExampleHiloUse {
	public static void main(String[] args) throws Exception {
		// BitcoinChartLOader.main(args);
		List<Integer> values = BitcoinRNNExampleHilo.loadValues();
		MultiLayerNetwork net = BitcoinRNNExampleHilo.load();

		int testCount = 10000;

		//findBestRandomized(values, net, testCount);
		
		 testAverageError(values, net, testCount, 33, 16);
		// findBest(values, net, testCount);
		// test(values, net, testCount, BitcoinRNNExampleHilo.sampleLangth,
		// BitcoinRNNExampleHilo.sampleLangth);
		// test(values, net, testCount, BitcoinRNNExampleHilo.sampleLangth,
		// BitcoinRNNExampleHilo.sampleLangth / 2);
		// test(values, net, testCount, BitcoinRNNExampleHilo.sampleLangth / 2,
		// BitcoinRNNExampleHilo.sampleLangth / 2);

	}

	private static void test(List<Integer> values, MultiLayerNetwork net, int testCount, int inputLength, int outputLength) {
		int rightCount = 0;
		for (int t = 0; t < testCount; t++) {
			boolean right = BitcoinRNNExampleHilo.testNetwork(values, net, inputLength, outputLength, false);
			if (right)
				rightCount++;
		}
		System.out.println("right " + ((double) rightCount / testCount));
	}

	private static void testAverageError(List<Integer> values, MultiLayerNetwork net, int testCount, int inputLength, int outputLength) {
		int error = 0;
		for (int t = 0; t < testCount; t++) {
			error += Math.abs(BitcoinRNNExampleHilo.testNetworkError(values, net, inputLength, outputLength, false));
		}
		System.out.println("average error " + ((double)error / testCount));
	}
	
	private static void findBestRandomized(List<Integer> values, MultiLayerNetwork net, int testCount) {

		double best = 0;
		int bestK = 0;
		int bestL = 0;

		for (int i = 0; i < 100; i++) {
			int k = (int) (1 + 96 * Math.random());
			int l = (int) (1 + 96 * Math.random());

			int rightCount = 0;
			for (int t = 0; t < testCount; t++) {
				boolean right = BitcoinRNNExampleHilo.testNetwork(values, net, k, l, false);
				if (right)
					rightCount++;
			}
			double ratio = (double) rightCount / testCount;
			System.out.println(k + "/" + l + " right " + ratio);
			if (ratio > best) {
				bestK = k;
				bestL = l;
				best = ratio;
				System.out.println("new record");
			}
		}
		System.out.println("best k " + bestK + " best l " + bestL + " reight " + best);
	}

	private static void findBest(List<Integer> values, MultiLayerNetwork net, int testCount) {

		double best = 0;
		int bestK = 0;
		int bestL = 0;

		for (int l = 5; l < 100; l++) {
			for (int k = 5; k < 100; k++) {
				int rightCount = 0;
				for (int t = 0; t < testCount; t++) {
					boolean right = BitcoinRNNExampleHilo.testNetwork(values, net, k, l, false);
					if (right)
						rightCount++;
				}
				double ratio = (double) rightCount / testCount;
				System.out.println(k + "/" + l + " right " + ratio);
				if (ratio > best) {
					bestK = k;
					bestL = l;
					best = ratio;
					System.out.println("new record");
				}
			}
		}
		System.out.println("best k " + bestK + " best l " + bestL + " reight " + best);
	}
}
