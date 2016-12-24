'''Images binary classifier based on scikit-learn SVM classifier.
It uses the RGB color space as feature vector.
'''

from __future__ import division
from __future__ import print_function
from PIL import Image, ImageTk
from sklearn import cross_validation
from sklearn import grid_search
from sklearn import svm
from sklearn import metrics
from StringIO import StringIO
from urlparse import urlparse
import urllib2
import sys
import os
import Tkinter as Tk
import tkFileDialog
from resizeimage import resizeimage
classifier = 5  
    
def process_directory(directory):
    '''Returns an array of feature vectors for all the image files in a
    directory (and all its subdirectories). Symbolic links are ignored.

    Args:
      directory (str): directory to process.

    Returns:
      list of list of float: a list of feature vectors.
    '''
    training = []
    for root, _, files in os.walk(directory):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            img_feature = process_image_file(file_path)
            if img_feature:
                training.append(img_feature)
    return training


def process_image_file(image_path):
    '''Given an image path it returns its feature vector.

    Args:
      image_path (str): path of the image file to process.

    Returns:
      list of float: feature vector on success, None otherwise.
    '''
    image_fp = StringIO(open(image_path, 'rb').read())
    try:
        image = Image.open(image_fp)
        return process_image(image)
    except IOError:
        return None

def show_usage():
    '''Prints how to use this program
    '''
    print("Usage: %s [class A images directory] [class B images directory]" %
            sys.argv[0])
    sys.exit(1)

def process_image_url(image_url):
    '''Given an image URL it returns its feature vector

    Args:
      image_url (str): url of the image to process.

    Returns:
      list of float: feature vector.

    Raises:
      Any exception raised by urllib2 requests.

      IOError: if the URL does not point to a valid file.
    '''
    parsed_url = urlparse(image_url)
    request = urllib2.Request(image_url)
    # set a User-Agent and Referer to work around servers that block a typical
    # user agents and hotlinking. Sorry, it's for science!
    request.add_header('User-Agent', 'Mozilla/5.0 (X11; Ubuntu; Linux ' \
            'x86_64; rv:31.0) Gecko/20100101 Firefox/31.0')
    request.add_header('Referrer', parsed_url.netloc)
    # Wrap network data in StringIO so that it looks like a file
    net_data = StringIO(urllib2.build_opener().open(request).read())
    size = 512, 512
    image = Image.open(net_data)
    image_resized = image.resize(size, Image.ANTIALIAS)
    return process_image(image_resized)
    
def process_image(image, blocks=4):
    '''Given a PIL Image object it returns its feature vector.
    
    Args:
    image (PIL.Image): image to process.
    blocks (int, optional): number of block to subdivide the RGB space into.
    
    Returns:
    list of float: feature vector if successful. None if the image is not
          RGB.
        '''
    if not image.mode == 'RGB':
        return None
    feature = [0] * blocks * blocks * blocks
    pixel_count = 0
    for pixel in image.getdata():
        ridx = int(pixel[0]/(256/blocks))
        gidx = int(pixel[1]/(256/blocks))
        bidx = int(pixel[2]/(256/blocks))
        idx = ridx + gidx * blocks + bidx * blocks * blocks
        feature[idx] += 1
        pixel_count += 1
    return [x/pixel_count for x in feature]

def train(training_path_a, training_path_b, training_path_c, training_path_d, training_path_e, training_path_f, print_metrics=False):
    global classifier
    '''Trains a classifier. training_path_a and training_path_b should be
    directory paths and each of them should not be a subdirectory of the other
    one. training_path_a and training_path_b are processed by
    process_directory().

    Args:
      training_path_a (str): directory containing sample images of class A.
      training_path_b (str): directory containing sample images of class B.
      print_metrics  (boolean, optional): if True, print statistics about
        classifier performance.

    Returns:
      A classifier (sklearn.svm.SVC).
    '''
    if not os.path.isdir(training_path_a):
        raise IOError('%s is not a directory' % training_path_a)
    if not os.path.isdir(training_path_b):
        raise IOError('%s is not a directory' % training_path_b)
    if not os.path.isdir(training_path_c):
        raise IOError('%s is not a directory' % training_path_c)
    if not os.path.isdir(training_path_d):
        raise IOError('%s is not a directory' % training_path_d)
    if not os.path.isdir(training_path_e):
        raise IOError('%s is not a directory' % training_path_e)
    if not os.path.isdir(training_path_f):
        raise IOError('%s is not a directory' % training_path_f)
    training_a = process_directory(training_path_a)
    training_b = process_directory(training_path_b)
    training_c = process_directory(training_path_c)
    training_d = process_directory(training_path_d)
    training_e = process_directory(training_path_e)
    training_f = process_directory(training_path_f)
    # data contains all the training data (a list of feature vectors)
    data = training_a + training_b + training_c + training_d + training_e + training_f
    # target is the list of target classes for each feature vector: a '1' for
    # class A and '0' for class B
    target = [0] * len(training_a) + [1] * len(training_b) + [2] * len(training_c) + [3] * len(training_d) + [4] * len(training_e) + [5] * len(training_f)
    # split training data in a train set and a test set. The test set will
    # containt 20% of the total
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(data,
            target, test_size=0.20)
    # define the parameter search space
    parameters = {'kernel': ['linear', 'rbf'], 'C': [1, 10, 100, 1000],
            'gamma': [0.01, 0.001, 0.0001]}
    # search for the best classifier within the search space and return it
    clf = grid_search.GridSearchCV(svm.SVC(), parameters).fit(x_train, y_train)
    classifier = clf.best_estimator_
    if print_metrics:
        print()
        print('Parameters:', clf.best_params_)
        print()
        print('Best classifier score')
        print(metrics.classification_report(y_test,
            classifier.predict(x_test)))
    return classifier

def initialize():
    global classifier
    '''Trains a classifier and allows to use it on images
    downloaded from the Internet.

    Args:
      training_path_a (str): directory containing sample images of class A.
      training_path_b (str): directory containing sample images of class B.
    '''
    training_path_a = 'C:/apples'
    training_path_b = 'C:/oranges'
    training_path_c = 'C:/bananas'
    training_path_d = 'C:/mangos'
    training_path_e = 'C:/grapes'
    training_path_f = 'C:/blueberries'
    
    classifier = train(training_path_a, training_path_b, training_path_c, training_path_d, training_path_e, training_path_f)

class FruitDetector:
    def __init__(self, master):
        self.master = master
        master.title("Fruit Detector")  
        root.geometry('{}x{}'.format(500,580))
        root.resizable(width=False, height=False)
        
        self.blank1 = Tk.Label()
        self.blank2 = Tk.Label()
        self.blank3 = Tk.Label()
        self.blank4 = Tk.Label()
        self.blank5 = Tk.Label()
        self.blank6 = Tk.Label()
        self.blank7 = Tk.Label()
        self.label = Tk.Label(master, text= "Fruits that can be detected:", font=("Helvetica", 16))
        self.list1 = Tk.Label(master, text = "Apple       Banana      Grape", font=("Helvetica", 11))       
        self.list2 = Tk.Label(root, text = "Grape       Blueberry       Mango", font=("Helvetica", 11))

        self.urlEntryVar = Tk.StringVar()
        self.entryLabel = Tk.Label(master, text="Provide URL", font=("Helvetica", 11))
        self.urlEntry = Tk.Entry(master, textvariable=self.urlEntryVar)
        
        self.broButton = Tk.Button(master = root, text = 'Browse', width = 6, command=self.process_file_input)

        
        
        self.submitButton = Tk.Button(master, text="Submit", command= self.process_user_input)
        
        self.outputFruit = ""
        self.outputText = Tk.StringVar()
        self.outputText.set(self.outputFruit)
        self.output = Tk.Label(master, textvariable=self.outputText, font=("Helvetica", 14))
        
        self.imageLabel = Tk.Label(master, border = 25)
        
        
        self.blank1.pack()
        self.label.pack()
        self.blank2.pack()
        self.list1.pack()
        self.list2.pack()
        self.blank3.pack()
        self.blank4.pack()
        self.entryLabel.pack()
        self.urlEntry.pack()
        self.blank5.pack()
        self.submitButton.pack()
        self.blank7.pack()
        self.broButton.pack()
        self.blank6.pack()
        self.output.pack()
        self.imageLabel.pack()
        
        
    def process_file_input(self):
        fname = tkFileDialog.askopenfilename(filetypes = (("Template files", "*.type"), ("All files", "*")))
        features = process_image_file(fname)
        
        self.outputFruit = self.get_fruit_name(classifier.predict(features))
        self.outputText.set(self.outputFruit)
        
        img = Image.open(fname)
        #img_resized = img.resize(size, Image.ANTIALIAS)
        img_resized = resizeimage.resize_contain(img, [256, 256])
        imgTk = ImageTk.PhotoImage(img_resized)
        self.imageLabel.configure(image = imgTk)
        self.imageLabel.image = imgTk
        
    def process_user_input(self):
        urlInput = self.urlEntry.get()
        self.urlEntry.delete(0, 'end')
        
        features = process_image_url(urlInput)
        
        #if file
        #features = self.process_image_file(pathInput)
        
    
        self.outputFruit = self.get_fruit_name(classifier.predict(features))
        self.outputText.set(self.outputFruit)
        
        # display image
        parsed_url = urlparse(urlInput)
        request = urllib2.Request(urlInput)
        # set a User-Agent and Referer to work around servers that block a typical
        # user agents and hotlinking. Sorry, it's for science!
        request.add_header('User-Agent', 'Mozilla/5.0 (X11; Ubuntu; Linux ' \
                'x86_64; rv:31.0) Gecko/20100101 Firefox/31.0')
        request.add_header('Referrer', parsed_url.netloc)
        # Wrap network data in StringIO so that it looks like a file
        net_data = StringIO(urllib2.build_opener().open(request).read())
        
        #size = 512, 512
        img = Image.open(net_data)
        #img_resized = img.resize(size, Image.ANTIALIAS)
        img_resized = resizeimage.resize_contain(img, [256, 256])
        imgTk = ImageTk.PhotoImage(img_resized)
        self.imageLabel.configure(image = imgTk)
        self.imageLabel.image = imgTk
        
        #panel.pack(side = "bottom", fill = "both", expand = "yes")
        ## end image display
        
        
        

    
    def get_fruit_name(self, int_val):
        if(int_val == 0):
            fruit = "APPLE"
        elif(int_val == 1):
            fruit = "ORANGE"
        elif(int_val == 2):
            fruit = "BANANA"
        elif(int_val == 3):
            fruit = "MANGO"
        elif(int_val == 4):
            fruit = "GRAPE"
        elif(int_val == 5):
            fruit = "BLUEBERRY"
        else:
            fruit = "other"
        return fruit
        
initialize() 
root = Tk.Tk()   
my_gui = FruitDetector(root)
root.mainloop()





