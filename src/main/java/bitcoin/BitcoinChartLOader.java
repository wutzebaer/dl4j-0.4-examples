package bitcoin;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Calendar;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.TimeZone;

import org.apache.commons.collections4.map.MultiKeyMap;
import org.apache.commons.io.IOUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.http.client.ClientProtocolException;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClientBuilder;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import org.json.JSONTokener;

public class BitcoinChartLOader {
	private static final int ZU_MERGENDE_ZEILEN = 30;
	private static final int RUNDEN_AUF_NACHKOMMASTELLEN = 1;

	public static void main(String[] args) throws ClientProtocolException, IOException, JSONException {
		createCoursesLog();
		createCleanBpi();
		mergeTimestamps();
		hilo();
	}

	private static void hilo() throws FileNotFoundException, IOException {

		BufferedReader is = new BufferedReader(new FileReader("courses_bpi_fixed_merged.log"));
		PrintWriter os = new PrintWriter(new BufferedWriter(new FileWriter("courses_bpi_fixed_merged_hilo.log")));

		// round to n nachkomma stellen
		double floor = Math.pow(10, RUNDEN_AUF_NACHKOMMASTELLEN);

		Double lastvalue = Math.floor(Double.valueOf(is.readLine().split(",")[1]) * floor) / floor;
		String line;
		while ((line = is.readLine()) != null) {
			String[] splits = line.split(",");
			Double currentvalue = Math.floor(Double.valueOf(splits[1]) * floor) / floor;
			Long time = Long.valueOf(splits[0]);
			os.println(time + "," + currentvalue.compareTo(lastvalue));
			lastvalue = currentvalue;
		}

		is.close();
		os.close();

	}

	private static void mergeTimestamps() throws FileNotFoundException, IOException {

		BufferedReader is = new BufferedReader(new FileReader("courses_bpi_fixed.log"));
		PrintWriter os = new PrintWriter(new BufferedWriter(new FileWriter("courses_bpi_fixed_merged.log")));

		// merge n minutes 1440=1tag
		final int mergeCount = ZU_MERGENDE_ZEILEN;

		double sum = 0;
		int mergedLines = 0;
		long mergeTime = 0;

		String line;
		while ((line = is.readLine()) != null) {
			String[] splits = line.split(",");
			Double value = Double.valueOf(splits[1]);
			Long time = Long.valueOf(splits[0]);
			if (mergedLines == 0) {
				mergeTime = time;
			}
			sum += value;
			mergedLines++;
			if (mergedLines == mergeCount) {
				os.println(mergeTime + "," + Math.round((sum / mergedLines) * 100d) / 100d);
				mergedLines = 0;
				sum = 0;
			}
		}

		// print rest
		if (mergedLines > 0) {
			os.println(mergeTime + "," + Math.round((sum / mergedLines) * 100d) / 100d);
		}

		is.close();
		os.close();

	}

	private static void createCleanBpi() throws FileNotFoundException, IOException {
		TimeZone.setDefault(TimeZone.getTimeZone("UTC"));

		BufferedReader is = new BufferedReader(new FileReader("courses.log"));
		PrintWriter os = new PrintWriter(new BufferedWriter(new FileWriter("courses_bpi_fixed.log")));
		String line;
		long lines = 0;
		int blindrow = 0;
		Double lastvalue = 0d;
		Long lasttime = 0l;
		// skip heads
		is.readLine();
		while ((line = is.readLine()) != null) {
			String[] splits = line.split(",");
			Double value = Double.valueOf(splits[1]);
			Long time = Long.valueOf(splits[0]);

			if (value == 0) {
				blindrow++;
			} else {
				if (blindrow > 0) {
					lasttime += 60000;
					double step = (value - lastvalue) / blindrow;
					while (lasttime < time) {
						lastvalue += step;
						os.println(lasttime + "," + Math.round(lastvalue * 100d) / 100d);
						lasttime += 60000;
					}
				}
				lastvalue = value;
				lasttime = time;
				blindrow = 0;
				os.println(line);
			}
			lines++;
		}
		is.close();
		os.close();
		System.out.println(lines);
	}

	private static void createCoursesLog() throws IOException, ClientProtocolException, FileNotFoundException, JSONException {
		TimeZone.setDefault(TimeZone.getTimeZone("UTC"));
		SimpleDateFormat format = new SimpleDateFormat("yyyy-MM-dd");
		CloseableHttpClient client = HttpClientBuilder.create().build();
		Calendar cal = Calendar.getInstance();

		// bpi,bitstamp,bitfinex,btce,coinbase,itbit,okcoin;
		// btce ist kaputt
		String providerlist = "bpi,okcoin";
		String[] providers = providerlist.split(",");

		Long time = 1451606400000L; // ab 2016
		// Long time = 1359261180000L; ab 16€
		// Long time = 1279411200000L;
		// Long time = 1459443420000L;

		// delete last two days
		{
			File tmpdir = new File("temp");
			tmpdir.mkdirs();
			File[] tmpfiles = tmpdir.listFiles();
			Arrays.sort(tmpfiles);
			for (int i = tmpfiles.length >= 2 ? tmpfiles.length - 2 : 0; i < tmpfiles.length; i++) {
				tmpfiles[i].delete();
			}
		}

		PrintWriter stream = new PrintWriter(new BufferedWriter(new FileWriter("courses.log")));

		HashSet<Long> loadedDates = new HashSet<Long>();

		stream.println("time," + StringUtils.join(providers, ","));

		MultiKeyMap loadedData = new MultiKeyMap();

		while (time < System.currentTimeMillis()) {
			HashMap<String, Double> allData = new HashMap<String, Double>();

			for (String provider : providers)
				allData.put(provider, 0.);

			// truncate time
			cal.setTimeInMillis(time);
			cal.set(Calendar.HOUR_OF_DAY, 0);
			cal.set(Calendar.MINUTE, 0);
			cal.set(Calendar.SECOND, 0);
			cal.set(Calendar.MILLISECOND, 0);
			final long date = cal.getTimeInMillis();

			// ensure data for date is loaded
			if (!loadedDates.contains(date)) {
				File tmpfile = new File("temp/" + date);

				loadedData.clear();

				if (!tmpfile.isFile()) {
					tmpfile.getParentFile().mkdirs();
					String dateStringFrom = format.format(cal.getTime());
					cal.add(Calendar.DATE, 1);
					String dateStringTo = format.format(cal.getTime());
					String url = "http://api.coindesk.com/charts/data?data=close&startdate=" + dateStringFrom + "&enddate=" + dateStringTo + "&exchanges=" + providerlist + "&dev=1&index=USD";
					// http://api.coindesk.com/charts/data?data=close&startdate=2016-03-30&enddate=2016-03-31&exchanges=bpi,okcoin&dev=1&index=USD
					CloseableHttpResponse response = client.execute(new HttpGet(url));
					InputStream is = response.getEntity().getContent();
					is.skip(3);
					FileOutputStream fos = new FileOutputStream(tmpfile);
					IOUtils.copy(is, fos);
					fos.close();
					is.close();
				}

				FileInputStream fis = new FileInputStream(tmpfile);
				JSONObject data = new JSONObject(new JSONTokener(new InputStreamReader(fis)));

				fis.close();

				// write loadedData
				for (String provider : providers) {
					if (data.has(provider)) {
						JSONArray providerArray = data.getJSONArray(provider);
						for (int i = 0; i < providerArray.length(); i++) {
							JSONArray providerValue = providerArray.getJSONArray(i);
							Long timestamp = providerValue.getLong(0);
							Double value = (providerValue.get(1).equals(JSONObject.NULL) ? 0 : providerValue.getDouble(1));
							loadedData.put(timestamp, provider, value);
						}
					}
				}

				loadedDates.add(date);

				System.out.println(new Date(date) + " " + date);
			}

			String line = "" + time;
			for (String provider : providers) {
				if (loadedData.containsKey(time, provider)) {
					allData.put(provider, (Double) loadedData.get(time, provider));
				}
			}

			for (String provider : providers) {
				line += "," + allData.get(provider);
			}

			stream.println(line);

			time += 60000;
		}

		stream.close();
	}
}
