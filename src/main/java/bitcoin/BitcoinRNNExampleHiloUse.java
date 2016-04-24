package bitcoin;

import java.util.List;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import bitcoin.BitcoinRNNExampleHilo.TestResult;

public class BitcoinRNNExampleHiloUse {
	public static void main(String[] args) throws Exception {
		// BitcoinChartLOader.main(args);
		List<Integer> values = BitcoinRNNExampleHilo.loadValues();
		MultiLayerNetwork net = BitcoinRNNExampleHilo.load();

		int testCount = 10000;

		// findBestRandomized(values, net, testCount);

		testPeaks(values, net, testCount, BitcoinRNNExampleHilo.sampleLangth, BitcoinRNNExampleHilo.sampleLangth / 2);
		// findBest(values, net, testCount);
		// test(values, net, testCount, BitcoinRNNExampleHilo.sampleLangth,
		// BitcoinRNNExampleHilo.sampleLangth);
		// test(values, net, testCount, BitcoinRNNExampleHilo.sampleLangth,
		// BitcoinRNNExampleHilo.sampleLangth / 2);
		// test(values, net, testCount, BitcoinRNNExampleHilo.sampleLangth / 2,
		// BitcoinRNNExampleHilo.sampleLangth / 2);

	}

	private static void testPeaks(List<Integer> values, MultiLayerNetwork net, int testCount, int inputLength, int outputLength) {

		int correctDetected = 0;
		int wrongAlert = 0;
		int missedDetection = 0;

		for (int t = 0; t < testCount; t++) {
			TestResult result = BitcoinRNNExampleHilo.testNetwork(values, net, inputLength, outputLength, false);
			if (result.expect > 10 && result.predict > 5 || result.expect < -10 && result.predict < -5) {
				correctDetected++;
			} else if (result.expect < 5 && result.predict > 10 || result.expect > -5 && result.predict < -10) {
				wrongAlert++;
			} else if ((result.expect > 10 || result.expect < -10)) {
				missedDetection++;
			}
		}

		System.out.println(String.format("correct:%s false:%s missed:%s", correctDetected, wrongAlert, missedDetection));

	}

}
