package Yelp.Evaluator

import Yelp.Trainer.NeuralNetwork._
import Yelp.Preprocessor.CSVImageMetadataReader._
import Yelp.Preprocessor.makeND4jDataSets.makeDataSetTE
import Yelp.Preprocessor.imageFeatureExtractor._
import Yelp.Evaluator.ResultFileGenerator._
import Yelp.Preprocessor.makeND4jDataSets._
import Yelp.Evaluator.ModelEvaluation._
import Yelp.Trainer.CNN._
import Yelp.Trainer.CNNEpochs._
import Yelp.Preprocessor.featureAndDataAligner

object p_explore_data {  
  val path4interpreter = "/Users/abrooks/Documents/github/kaggle_yelp/"
  
  val labMap = readBusinessLabels(path4interpreter + "data/labels/train.csv")
  val bizMap = readBusinessToImageLabels(path4interpreter + "data/labels/train_photo_to_biz_ids.csv")
  val imgs = getImageIds(path4interpreter + "data/images/train", bizMap, bizMap.map(_._2).toSet.toList.slice(1500,1515))
  val dataMap = processImages(imgs, resizeImgDim = 64) // nPixels = 64
  val alignedData = new featureAndDataAligner(dataMap, bizMap, Option(labMap))()
  
  bizMap.keys  
  
  val dsTE = makeDataSetTE(alignedData)
  val model = loadNN(path4interpreter + "results/modelsV0/model2_img16k_epoch15_batch128_pixels64_nout100_200.json", path4interpreter + "results/modelsV0/model2_img16k_epoch15_batch128_pixels64_nout100_200.bin")
  val predsTE = scoreModel(model, dsTE)
  
  val bizScoreAgg = aggImgScores2Business(predsTE, alignedData)
  println(bizScoreAgg)
  println(alignedData.getImgCntsPerBusiness)
  println(alignedData.getBusinessLabels)  
}