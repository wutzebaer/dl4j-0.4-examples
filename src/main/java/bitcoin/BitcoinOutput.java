package bitcoin;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.json.JSONException;

public class BitcoinOutput {
	public static void main(String[] args) throws JSONException, IOException {
		BufferedReader is = new BufferedReader(new FileReader("courses_bpi_fixed.log"));
		String line;
		List<Double> values = new ArrayList<>();
		while ((line = is.readLine()) != null) {
			String[] splits = line.split(",");
			values.add(Double.valueOf(splits[1]));
		}
		System.out.println(values.size());
	}
}
