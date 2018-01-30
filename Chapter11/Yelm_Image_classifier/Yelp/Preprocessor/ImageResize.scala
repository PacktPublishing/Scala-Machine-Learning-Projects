package Yelp.Preprocessor

import org.imgscalr._
import java.io.File
import javax.imageio.ImageIO

object ImageResize {
  def main(args: Array[String]): Unit = {

    def resizeImg(img: java.awt.image.BufferedImage, width: Int, height: Int) = {
      Scalr.resize(img, Scalr.Method.BALANCED, width, height)
    }

    val testImage = ImageIO.read(new File("data/images/train/147.jpg"))

    val testImage32 = resizeImg(testImage, 32, 32)
    val testImage64 = resizeImg(testImage, 64, 64)
    val testImage128 = resizeImg(testImage, 128, 128)
    val testImage256 = resizeImg(testImage, 256, 256)

    ImageIO.write(testImage32, "jpg", new File("data/images/preprocessed/147resize32.jpg"))
    ImageIO.write(testImage64, "jpg", new File("data/images/preprocessed/147resize64.jpg"))
    ImageIO.write(testImage128, "jpg", new File("data/images/preprocessed/147resize128.jpg"))
    ImageIO.write(testImage256, "jpg", new File("data/images/preprocessed/147resize256.jpg"))
  }
}