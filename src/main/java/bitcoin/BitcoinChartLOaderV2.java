package bitcoin;

import java.io.InputStream;

import org.apache.http.client.methods.HttpGet;
import org.apache.http.impl.client.HttpClientBuilder;
import org.json.JSONArray;
import org.json.JSONObject;
import org.json.JSONTokener;

public class BitcoinChartLOaderV2 {
	public static void main(String[] args) throws Exception {
		while (true) {
			JSONObject cUrrentPrice = getCUrrentPrice();
			System.out.println(cUrrentPrice);
			System.out.println(cUrrentPrice.getLong("server_time")*1000);
			System.out.println(cUrrentPrice.getDouble("last"));
			Thread.sleep(1000);
		}
	}

	public static JSONObject getCUrrentPrice() throws Exception {
		InputStream dataStream = HttpClientBuilder.create().build().execute(new HttpGet("https://btc-e.com/api/2/btc_usd/ticker")).getEntity().getContent();
		JSONObject array = new JSONObject(new JSONTokener(dataStream)).getJSONObject("ticker");
		dataStream.close();
		return array;
	}
}
