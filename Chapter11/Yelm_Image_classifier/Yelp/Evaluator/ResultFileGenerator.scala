package Yelp.Evaluator

import java.io.File
import Yelp.Trainer.NeuralNetwork._
import Yelp.Preprocessor.CSVImageMetadataReader._
import Yelp.Preprocessor.makeND4jDataSets.makeDataSetTE
import Yelp.Preprocessor.featureAndDataAligner
import Yelp.Preprocessor.imageFeatureExtractor._
import Yelp.Evaluator.ResultFileGenerator._
import Yelp.Preprocessor.makeND4jDataSets._
import Yelp.Evaluator.ModelEvaluation._
import Yelp.Trainer.CNN._
import Yelp.Trainer.CNNEpochs._
import scala.Vector

object ResultFileGenerator {
  def writeSubmissionFile(outcsv: String, phtoObj: List[(String, Vector[Double])], thresh: Double): Unit = {
    // prints to a csv or other txt file
    def printToFile(f: java.io.File)(op: java.io.PrintWriter => Unit) {
      val p = new java.io.PrintWriter(f)
      try { op(p) } finally { p.close() }
    }
    // assigning cutoffs for each class
    def findIndicesAboveThresh(x: Vector[Double]): Vector[Int] = {
      x.zipWithIndex.filter(x => x._1 >= thresh).map(_._2)
    }
    // create vector of rows to write to csv
    val ret = (for (i <- 0 until phtoObj.length) yield {
      (phtoObj(i)._1 + "," + findIndicesAboveThresh(phtoObj(i)._2).mkString(" "))
    }).toVector
    // actually write text file
    printToFile(new File(outcsv)) {
      p => (Vector("business_ids,labels") ++ ret).foreach(p.println)
    }
  }
  /** Create csv to submit to kaggle with predicted classes from all 9 categories for each business in the test image set */
  def SubmitObj(alignedData: featureAndDataAligner,
    modelPath: String,
    model0: String = "model0",
    model1: String = "model1",
    model2: String = "model2",
    model3: String = "model3",
    model4: String = "model4",
    model5: String = "model5",
    model6: String = "model6",
    model7: String = "model7",
    model8: String = "model8"): List[(String, Vector[Double])] = {

    // new code which works in REPL    
    // creates a List for each model (class) containing a map from the bizID to the probability of belonging in that class 
    val big = for (m <- List(model0, model1, model2, model3, model4, model5, model6, model7, model8)) yield {
      val ds = makeDataSetTE(alignedData)
      val model = loadNN(modelPath + m + ".json", modelPath + m + ".bin")
      val scores = scoreModel(model, ds)
      val bizScores = aggImgScores2Business(scores, alignedData)
      bizScores.toMap
    }

    // transforming the data structure above into a List for each bizID containing a Tuple (bizid, List[Double]) where the Vector[Double] is the 
    // the vector of probabilities 
    alignedData.data.map(_._2).distinct map (x =>
      (x, big.map(x2 => x2(x)).toVector))
  }
}