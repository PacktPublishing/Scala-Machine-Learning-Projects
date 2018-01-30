package Yelp.Preprocessor

import scala.io.Source

object CSVImageMetadataReader {  
  /** Generic function to read image metadata */  
  def readMetadata(csv: String, rows: List[Int]=List(-1)):  List[List[String]] = {
    val src = Source.fromFile(csv) 
    def reading(csv: String): List[List[String]] = {  
      src.getLines.map(x => x.split(",").toList)
         .toList
    }
    //src.close
    try {
        if(rows==List(-1)) reading(csv)
        else rows.map(reading(csv))
    } finally {
        src.close
    }        
  }  
  
  /** Create map from bizid to labels of form bizid -> Set(labels)  */  
  def readBusinessLabels(csv: String, rows: List[Int]=List(-1)): Map[String, Set[Int]]  = {
    val reader = readMetadata(csv)
    reader.drop(1) // should make this conditional or handle in pattern-matching
       .map(x => x match {
          case x :: Nil => (x(0).toString, Set[Int]())
          case _ => (x(0).toString, x(1).split(" ").map(y => y.toInt).toSet)
          }).toMap
  }
  
  /** Create map from imgID to bizID of form imgID -> busID  */    
  def readBusinessToImageLabels(csv: String, rows: List[Int] = List(-1)): Map[Int, String]  = {
    val reader = readMetadata(csv)
    reader.drop(1)
       .map(x => x match {
         case x :: Nil => (x(0).toInt, "-1")
          case _ => (x(0).toInt, x(1).split(" ").head)
       }).toMap
  }  
}