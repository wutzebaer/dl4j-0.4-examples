package bitcoin;

public class ConfigLine {
	public int historycount;
	public int futurecount;
	public double minPlus;
	public int hiddenLayerCount;
	public int hiddenLayerWidth;
	public int samplesPerDataSet;
	public int iterations;
	public double weightA;
	public double weightB;
	public Double success;
	public Integer positivesExpected;
	public Integer foundPositived;
	public Integer falsePositives;

	public ConfigLine(String line) {
		String[] bits = line.split(" ");
		success = Double.valueOf(bits[0].split(":")[1].replace(",", "."));
		positivesExpected = Integer.valueOf(bits[1].split(":")[1]);
		foundPositived = Integer.valueOf(bits[2].split(":")[1]);
		falsePositives = Integer.valueOf(bits[3].split(":")[1]);

		historycount = Integer.valueOf(bits[4].split(":")[1]);
		futurecount = Integer.valueOf(bits[5].split(":")[1]);
		minPlus = Double.valueOf(bits[6].split(":")[1].replace(",", "."));
		hiddenLayerCount = Integer.valueOf(bits[7].split(":")[1]);
		hiddenLayerWidth = Integer.valueOf(bits[8].split(":")[1]);
		samplesPerDataSet = Integer.valueOf(bits[9].split(":")[1]);
		iterations = Integer.valueOf(bits[10].split(":")[1]);
		weightA = Double.valueOf(bits[11].split(":")[1].replace(",", "."));
		weightB = Double.valueOf(bits[12].split(":")[1].replace(",", "."));
	}

}