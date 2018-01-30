package Yelp.Preprocessor

class featureAndDataAligner(dataMap: Map[Int, Vector[Int]], bizMap: Map[Int, String], labMap: Option[Map[String, Set[Int]]])(rowindices: List[Int] = dataMap.keySet.toList) {  
  // initializing alignedData with empty labMap when it is not provided (we are working with training data)
  def this(dataMap: Map[Int, Vector[Int]], bizMap: Map[Int, String])(rowindices: List[Int]) = this(dataMap, bizMap, None)(rowindices)

  def alignBusinessImgageIds(dataMap: Map[Int, Vector[Int]], bizMap: Map[Int, String])
    (rowindices: List[Int] = dataMap.keySet.toList): List[(Int, String, Vector[Int])] = {
      for { pid <- rowindices
          val imgHasBiz = bizMap.get(pid) // returns None if img does not have a bizID
          val bid = if(imgHasBiz != None) imgHasBiz.get else "-1"
          if (dataMap.keys.toSet.contains(pid) && imgHasBiz != None)
      } yield { 
          (pid, bid, dataMap(pid))
      }
  }
  
  def alignLabels(dataMap: Map[Int, Vector[Int]], bizMap: Map[Int, String], labMap: Option[Map[String, Set[Int]]])
    (rowindices: List[Int] = dataMap.keySet.toList): List[(Int, String, Vector[Int], Set[Int])] = {
      def flatten1[A, B, C, D](t: ((A, B, C), D)): (A, B, C, D) = (t._1._1, t._1._2, t._1._3, t._2)
      val al = alignBusinessImgageIds(dataMap, bizMap)(rowindices)
      for { p <- al
      } yield {
        val bid = p._2
        val labs = labMap match  {
          case None => Set[Int]()
          case x => (if(x.get.keySet.contains(bid)) x.get(bid) else Set[Int]())
        }
        flatten1(p, labs) 
      }
  }
  
  // pre-computing and saving data as a val so method does not need to re-compute each time it is called. 
  lazy val data = alignLabels(dataMap, bizMap, labMap)(rowindices)
  
  // getter functions
  def getImgIds = data.map(_._1)
  def getBusinessIds = data.map(_._2)
  def getImgVectors = data.map(_._3)
  def getBusinessLabels = data.map(_._4)  
  def getImgCntsPerBusiness = getBusinessIds.groupBy(identity).mapValues(x => x.size)   
}