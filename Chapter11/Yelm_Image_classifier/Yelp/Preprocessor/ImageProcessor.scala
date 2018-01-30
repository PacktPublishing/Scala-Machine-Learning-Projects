package Yelp.Preprocessor

import scala.Vector
import org.imgscalr._

object imageUtils {
   implicit class imageProcessingPipeline(img: java.awt.image.BufferedImage) {    
    // image 2 vector processing
    def pixels2gray(red: Int, green:Int, blue: Int): Int = (red + green + blue) / 3
    def pixels2color(red: Int, green:Int, blue: Int): Vector[Int] = Vector(red, green, blue)
  
    private def image2vec[A](f: (Int, Int, Int) => A ): Vector[A] = {
               val w = img.getWidth
               val h = img.getHeight
               for { w1 <- (0 until w).toVector
                     h1 <- (0 until h).toVector
                   } yield {
                       val col = img.getRGB(w1, h1)
        			         val red =  (col & 0xff0000) / 65536
        			         val green = (col & 0xff00) / 256
        			         val blue = (col & 0xff)
                       f(red, green, blue)
                   }
             }
    
    def image2gray: Vector[Int] = image2vec(pixels2gray)
    def image2color: Vector[Int] = image2vec(pixels2color).flatten
    
    // make image square
    def makeSquare = {
      val w = img.getWidth
      val h = img.getHeight
      val dim = List(w, h).min
      
      img match {
    	  case x if w == h => img
    	  case x if w > h => Scalr.crop(img, (w-h)/2, 0, dim, dim)
    	  case x if w < h => Scalr.crop(img, 0, (h-w)/2, dim, dim)
      }
    }                        
    
    // resize pixels
    def resizeImg(width: Int, height: Int) = {
      Scalr.resize(img, Scalr.Method.BALANCED, width, height)
    }    
   }
  }