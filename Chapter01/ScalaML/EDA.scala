package com.packt.ScalaML

object EDA {
  def main(args: Array[String]): Unit = {
    val spark = SparkSessionCreate.createSession()
    import spark.implicits._
    
    val df = Preproessing.trainInput   
    df.show()
    
    //Let's show some seected column only. But feel free to use DF.show() to see the all columns. 
    df.select("id", "cat1", "cat2", "cat3", "cont1", "cont2", "cont3", "loss").show()
    
    //If you see all the rows sing df.show() you will see some categorical columns contains too many categories. 
    df.select("cat109", "cat110", "cat112", "cat113", "cat116").show()

    println(df)
    print(df.printSchema())
    
    val newDF = df.withColumnRenamed("loss", "label")
    newDF.createOrReplaceTempView("insurance")
    spark.sql("SELECT avg(insurance.label) as AVG_LOSS FROM insurance").show()
    spark.sql("SELECT min(insurance.label) as MIN_LOSS FROM insurance").show()
    spark.sql("SELECT max(insurance.label) as MAX_LOSS FROM insurance").show()
  }
}