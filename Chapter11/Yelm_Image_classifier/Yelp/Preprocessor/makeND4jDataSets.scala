package Yelp.Preprocessor

import org.nd4j.linalg.dataset.{DataSet}
import org.nd4s.Implicits._ // should work with nd4s package
import org.nd4j.linalg.api.ndarray.INDArray
import featureAndDataAligner._

object makeND4jDataSets {  
  /** Creates DataSet object from the data structure of data structure from alignLables function in form List[(imgID, bizID, labels, pixelVector)]  */
    def makeDataSet(alignedData: featureAndDataAligner, bizClass: Int): DataSet = {
    val alignedXData = alignedData.getImgVectors.toNDArray
    val alignedLabs = alignedData.getBusinessLabels.map(x => if (x.contains(bizClass)) Vector(1, 0) else Vector(0, 1)).toNDArray
    new DataSet(alignedXData, alignedLabs) 
  }
  
  def makeDataSetTE(alignedData: featureAndDataAligner): INDArray = {
    alignedData.getImgVectors.toNDArray
  }  
}