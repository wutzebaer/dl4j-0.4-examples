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
import java.util.HashMap;
import java.util.List;

import org.apache.commons.codec.digest.DigestUtils;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.StringUtils;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

public class BitcoinFeedForwardUser {
	public static void main(String[] args) throws Exception {
		List<String> configLines = new ArrayList<>();

		for (File f : new File(".").listFiles()) {
			if (f.getName().endsWith("_records.txt")) {
				BufferedReader bufferedReader = new BufferedReader(new FileReader(f));
				String line;
				while ((line = bufferedReader.readLine()) != null) {
					// ignore empty lines
					if (StringUtils.isBlank(line)) {
						continue;
					}
					configLines.add(line);
				}
				bufferedReader.close();
			}
		}

		BitcoinFeedForward.initValues();

		HashMap<String, MultiLayerNetwork> networks = new HashMap<>();

		// train and save networks
		for (String line : configLines) {
			ConfigLine configLine = new ConfigLine(line);

			if (configLine.foundPositived < 100 || configLine.success < 0.8) {
				continue;
			}

			if (!new File("nets/" + DigestUtils.md5Hex(line) + "_coefficients.bin").isFile()) {
				MultiLayerNetwork network = BitcoinFeedForward.testConfigString(configLine).network;
				SaveLoad.save(network, line);
			} else {
				System.out.println("exists " + line);
			}
			networks.put(line, SaveLoad.load(line));
		}

		System.out.println("Loaded " + networks.size());

		// load newest charts
		BitcoinChartLOader.main(args);
		BitcoinFeedForward.initValues();

		int yes = 0;
		int no = 0;
		// let every network vote
		for (MultiLayerNetwork n : networks.values()) {
			int historycount = ((FeedForwardLayer) n.getLayer(0).conf().getLayer()).getNIn();
			INDArray input = Nd4j.zeros(1, historycount);
			int startindex = BitcoinFeedForward.values.size() - historycount;
			double firstvalue = BitcoinFeedForward.values.get(startindex);
			for (int i = 0; i < historycount; i++) {
				input.putScalar(new int[] { 0, i }, BitcoinFeedForward.values.get(startindex + i) - firstvalue);
			}
			INDArray output = n.output(input);
			if (output.getDouble(0, 0) > 0.9) {
				yes++;
			} else {
				no++;
			}
		}
		System.out.println(String.format("Yes:%d no:%d", yes, no));

		for (int j = 0; j < 1000; j++) {
			yes = 0;
			no = 0;
			int historycount = BitcoinFeedForward.maxRandomHistoryCount;
			int futurecount = BitcoinFeedForward.maxRandomFutureCount;
			DataSet ds = BitcoinFeedForward.createDataset(1, historycount, futurecount, BitcoinFeedForward.minPlus);
			INDArray inputRow = ds.getFeatureMatrix().getRow(0);

			// let every network vote
			for (MultiLayerNetwork n : networks.values()) {
				int networkHistoryCount = ((FeedForwardLayer) n.getLayer(0).conf().getLayer()).getNIn();

				INDArray input = Nd4j.zeros(1, networkHistoryCount);
				for (int s = 0; s < networkHistoryCount; s++) {
					input.putScalar(s, inputRow.getDouble(inputRow.length() - networkHistoryCount + s));
				}

				INDArray output = n.output(input);
				if (output.getDouble(0, 0) > 0.9) {
					yes++;
				} else {
					no++;
				}
			}
			if (yes > networks.size() / 2)
				System.out.println(String.format("Yes:%d no:%d expect:%d", yes, no, ds.getLabels().getInt(0, 0)));
		}

	}
	
}
