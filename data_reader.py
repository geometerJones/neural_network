
class Image:
  def __init__(self, label):
    self.pixels = []
    self.label = label

class DataReader:
  @staticmethod
  def GetImages(imagefilename, labelfilename):
    """Returns a list of image objects
    imagefilename: The file to read in that holds the images
    labelfilename: The file to read in that holds the labels
    """
    
    images = []
    imagefile = open(imagefilename, 'r')
    labelfile = open(labelfilename, 'r')
    image = None
    
    while True:
      imageline = imagefile.readline().strip(',')
      labelline = labelfile.readline().strip()
      if not (imageline and labelline):
        break
      image = Image(int(labelline[0]))
      image.pixels.append([float(r) for r in imageline.strip().split()])
      images.append(image)
    return images
  
  def GetSortedImages(n_filename, p_filename):
    """Returns a list of image objects
    imagefilename: The file to read in that holds the images
    labelfilename: The file to read in that holds the labels
    """
    
    images = []
    p_file = open(p_filename, 'r')
    n_file = open(n_filename, 'r')
    
    while True:
      p_line = p_file.readline().strip('()')
      n_line = n_file.readline().strip('()')
      if not (p_line or n_line):
        break
       if n_line:
        image = Image(1)
        image.pixels.append([float(r) for r in n_line.strip().split(', ')])
        images.append(image)
      if p_line:
        image = Image(0)
        image.pixels.append([float(r) for r in p_line.strip().split(', ')])
        images.append(image)

    return images
    
  @staticmethod
  def DumpWeights(weights, filename):
    """Dump the weights vector to filename"""
    
    outfile = open(filename, 'w')
    for weight in weights:
      outfile.write('%r\n' % weight)

  @staticmethod
  def ReadWeights(filename):
    """Returns a weight vector retrieved by reading file filename"""
    
    infile = open(filename, 'r')
    weights = []
    for line in infile:
      weight = float(line.strip())
      weights.append(weight)
    return weights