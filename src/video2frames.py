import cv2
# from skimage.transform import resize
import sys

def main():
    if len(sys.argv) != 3:
        raise Exception('usage: python project1.py <video.mp4> <output folder>')

    infile, outfolder = sys.argv[1], sys.argv[2]
    filename = infile[infile.rfind('/') + 1:infile.find('.mp4')]
    vidcap = cv2.VideoCapture(infile)
    success, image = vidcap.read()
    count = 0
    while success:
      # image = resize(image, (144, 256), mode='constant', anti_aliasing=True) * 256
      cv2.imwrite('{}/{}-{}.jpg'.format(outfolder, filename, count), image)  
      success,image = vidcap.read()
      count += 1
      if not success:
        print('Failed reading frame {}'.format(count))

if __name__ == '__main__':
    main()
