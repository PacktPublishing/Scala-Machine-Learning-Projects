package Yelp.Classifier

import Yelp.Preprocessor.CSVImageMetadataReader._
import Yelp.Preprocessor.featureAndDataAligner
import Yelp.Preprocessor.imageFeatureExtractor._
import Yelp.Evaluator.ResultFileGenerator._
import Yelp.Preprocessor.makeND4jDataSets._
import Yelp.Evaluator.ModelEvaluation._
import Yelp.Trainer.CNNEpochs._
import Yelp.Trainer.NeuralNetwork._

object YelpImageClassifier {
  def main(args: Array[String]): Unit = {    
  // image processing on training data
  val labelMap = readBusinessLabels("data/labels/train.csv")
  val businessMap = readBusinessToImageLabels("data/labels/train_photo_to_biz_ids.csv")
  val imgs = getImageIds("data/images/train/", businessMap, businessMap.map(_._2).toSet.toList).slice(0,100) // 20000 images
  println("Image ID retreival done!")
  
  val dataMap = processImages(imgs, resizeImgDim = 256) 
  println("Image processing done!")
  
  val alignedData = new featureAndDataAligner(dataMap, businessMap, Option(labelMap))()
  println("Feature extraction done!")
  
  // training one model for one class at a time. Many hyperparamters hardcoded within 
  val cnn0 = trainModelEpochs(alignedData, businessClass = 0, saveNN = "models/model0") 
  val cnn1 = trainModelEpochs(alignedData, businessClass = 1, saveNN = "models/model1") 
  val cnn2 = trainModelEpochs(alignedData, businessClass = 2, saveNN = "models/model2")
  val cnn3 = trainModelEpochs(alignedData, businessClass = 3, saveNN = "models/model3")
  val cnn4 = trainModelEpochs(alignedData, businessClass = 4, saveNN = "models/model4")
  val cnn5 = trainModelEpochs(alignedData, businessClass = 5, saveNN = "models/model5")
  val cnn6 = trainModelEpochs(alignedData, businessClass = 6, saveNN = "models/model6")
  val cnn7 = trainModelEpochs(alignedData, businessClass = 7, saveNN = "models/model7")
  val cnn8 = trainModelEpochs(alignedData, businessClass = 8, saveNN = "models/model8")

  // processing test data for scoring
  val businessMapTE = readBusinessToImageLabels("data/labels/test_photo_to_biz.csv")
  val imgsTE = getImageIds("data/images/test//", businessMapTE, businessMapTE.map(_._2).toSet.toList)
  val dataMapTE = processImages(imgsTE, resizeImgDim = 128) // make them 256x256
  val alignedDataTE = new featureAndDataAligner(dataMapTE, businessMapTE, None)()
  
  // creating csv file to submit to kaggle (scores all models)
  val Results = SubmitObj(alignedDataTE, "results/ModelsV0/")
  val SubmitResults = writeSubmissionFile("results/kaggleSubmission/kaggleSubmitFile.csv", Results, thresh = 0.9)
  }
}