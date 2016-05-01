package bitcoin;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.file.Files;
import java.nio.file.Paths;

import org.apache.commons.codec.digest.DigestUtils;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * @author Peter Gro�mann
 */
public class SaveLoad {

	public static void save(MultiLayerNetwork net, String filename) throws IOException {
		filename = DigestUtils.md5Hex(filename);

		new File("nets").mkdirs();

		// Write the network parameters:
		try (DataOutputStream dos = new DataOutputStream(Files.newOutputStream(Paths.get("nets/" + filename + "_coefficients.bin")))) {
			Nd4j.write(net.params(), dos);
		}

		// Write the network configuration:
		FileUtils.write(new File("nets/" + filename + "_conf.json"), net.getLayerWiseConfigurations().toJson());

		// Save the updater:
		try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream("nets/" + filename + "_updater.bin"))) {
			oos.writeObject(net.getUpdater());
		}
	}

	public static MultiLayerNetwork load(String filename) throws FileNotFoundException, IOException, ClassNotFoundException {
		filename = DigestUtils.md5Hex(filename);
		// Load network configuration from disk:
		MultiLayerConfiguration confFromJson = MultiLayerConfiguration.fromJson(FileUtils.readFileToString(new File("nets/" + filename + "_conf.json")));

		// Load parameters from disk:
		INDArray newParams;
		try (DataInputStream dis = new DataInputStream(new FileInputStream("nets/" + filename + "_coefficients.bin"))) {
			newParams = Nd4j.read(dis);
		}

		// Create a MultiLayerNetwork from the saved configuration and
		// parameters
		MultiLayerNetwork savedNetwork = new MultiLayerNetwork(confFromJson);
		savedNetwork.init();
		// must be set before setUpdater()
		savedNetwork.setParameters(newParams);

		// Load the updater:
		org.deeplearning4j.nn.api.Updater updater;
		try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream("nets/" + filename + "_updater.bin"))) {
			updater = (org.deeplearning4j.nn.api.Updater) ois.readObject();
		}

		// Set the updater in the network
		savedNetwork.setUpdater(updater);

		return savedNetwork;
	}
}