import cv2

def main():
    if len(sys.argv) != 3:
        raise Exception('usage: python project1.py <video.mp4> <output folder>')

    infile, outfolder = sys.argv[1], sys.argv[2]
    filename = infile[:infile.find('.mp4')]
    vidcap = cv2.VideoCapture(infile)
    success,image = vidcap.read()
    count = 0
    while success:
      cv2.imwrite('{}-{}.jpg'.format(filename, count), image)  
      success,image = vidcap.read()
      print('Failed reading frame {}'.format(count))
      count += 1

if __name__ == '__main__':
    main()
