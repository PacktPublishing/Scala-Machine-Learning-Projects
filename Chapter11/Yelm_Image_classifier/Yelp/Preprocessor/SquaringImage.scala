package Yelp.Preprocessor

import org.imgscalr._
import java.io.File
import javax.imageio.ImageIO

object SquaringImage {
  def main(args: Array[String]): Unit = {
    def makeSquare(img: java.awt.image.BufferedImage): java.awt.image.BufferedImage = {
      val w = img.getWidth
      val h = img.getHeight
      val dim = List(w, h).min

      img match {
        case x if w == h => img
        case x if w > h => Scalr.crop(img, (w - h) / 2, 0, dim, dim)
        case x if w < h => Scalr.crop(img, 0, (h - w) / 2, dim, dim)
      }
    }

    val myimg = ImageIO.read(new File("data/images/train/147.jpg"))
    val myimgSquare = makeSquare(myimg)
    ImageIO.write(myimgSquare, "jpg", new File("data/images/preprocessed/147square.jpg"))
  }
}