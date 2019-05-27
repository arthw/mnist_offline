import gzip
import pickle
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import cv2

def resize(img, scale=10.0 ):
  width = int(img.shape[1] * scale)
  height = int(img.shape[0] * scale)
  dim = (width, height)
  resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 
  return resized
  
def show_train_pic(data_sets):
  print("Total Images: %d" % len(data_sets.train.images))
  images = data_sets.train.images
  images_t = data_sets.test.images
  for i in range(len(data_sets.train.images)):
    cv2.imshow('Train', resize(images[i].reshape((28, 28)), 10.0))
    cv2.imshow('Test', resize(images_t[i].reshape((28, 28)), 10.0))
    #cv2.waitKey(0)
    if cv2.waitKey(50) & 0xFF == ord('q') or i>100: 
        break
  print('Show done')
  cv2.destroyAllWindows()
  print('Exit Window!')

  #plt.imshow(images[0].reshape((28, 28)), cmap=cm.Greys_r)
  #plt.show()


if __name__=="__main__":
  data_sets = input_data.read_data_sets("")
  show_train_pic(data_sets)