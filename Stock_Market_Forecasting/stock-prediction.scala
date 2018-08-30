import org.apache.spark.SparkContext
import spark.implicits._
import org.apache.spark.sql
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import com.cloudera.sparkts.DateTimeIndex
import com.cloudera.sparkts._
import java.time._
import com.cloudera.sparkts.models._
import org.apache.spark.mllib.linalg._
import org.apache.spark.sql.SparkSession
/* As per requirement, we are having training Dataset have to predict next day close value and open value
*/

// lets predict Stock close price 

def main(args: Array[String]) {
	
val spark = SparkSession.builder().appName("Spark--Stock Price Forcast Prediction").getOrCreate()


val input_data = spark.read.format("CSV").option("inferSchema",true).option("header",true).load(args(0))
val closeStockInputDf = input_data.select(unix_timestamp($"date", "MM/dd/yyyy").cast(TimestampType).as("timestamp"),$"symbol",$"close").withColumnRenamed("close","price")

val minDate= closeStockInputDf.selectExpr("min(timestamp)").collect()(0).getTimestamp(0)
val maxDate = closeStockInputDf.selectExpr("max(timestamp)").collect()(0).getTimestamp(0)
val zone = ZoneId.systemDefault()

val dateIndex = DateTimeIndex.uniformFromInterval(ZonedDateTime.of(minDate.toLocalDateTime, zone), ZonedDateTime.of(maxDate.toLocalDateTime, zone),new DayFrequency(1))

val DAYS=1
val timeseriesRdd = TimeSeriesRDD.timeSeriesRDDFromObservations(dateIndex,closeStockInputDf,"timestamp", "symbol", "price")
val FinalResult = timeseriesRdd.mapSeries{vector => {
  val newVec = new DenseVector(vector.toArray.map(x => if(x.equals(Double.NaN)) 0 else x))
  val arimaModel = ARIMA.fitModel(1, 0, 0, newVec)
  val forecasted = arimaModel.forecast(newVec, DAYS)
  new DenseVector(forecasted.toArray.slice(forecasted.size-(DAYS+1), forecasted.size-1))
 }}.toDF("symbol","values")
FinalResult.write.save(args(1))
}