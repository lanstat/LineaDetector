import cv2
import numpy as np
import sys
import argparse
import json

parser = argparse.ArgumentParser(description="Get director vector from a image or a video")
parser.add_argument('filename', help='name of the file to process')
parser.add_argument('-d', '--dump', dest='dump', nargs='?', const=True, default=False,
                    help='write the images generated in the process')
parser.add_argument('-ml', '--min-length', dest='min_line_length', nargs='?', default=35,
                    help='minimum line length the be accepted, the value is a percentage' +
                    ' of the image width (default: 35)', type=float)
parser.add_argument('-tls', '--threshold-line-size', dest='threshold_line_size', nargs='?', default=6.25,
                    help='limit of line width that will be used to unify the found vectors,' +
                    ' the value is a percentage of the image width (default: 6.25)', type=float)

args = parser.parse_args()

filename = args.filename

def dump_image(original, red_lines, lines, avg_lines):
    line_image = np.copy(original)
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)

    if red_lines is not None:
        for line in red_lines:
            for x1,y1,x2,y2 in line:
                cv2.line(line_image,(x1,y1),(x2,y2),(255,255,0),5)

    if avg_lines is not None:
        for line in avg_lines:
            cv2.line(line_image,(line['x1'],line['y1']),(line['x2'],line['y2']),
                     (255,0,255),5)

    lines_edges = cv2.addWeighted(original, 0.8, line_image, 1, 0)
    cv2.imshow(filename, line_image)
    cv2.waitKey()

def process():
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    if img is None:
        print('file doesnt exists')
        sys.exit()

    min_line_length = img.shape[0] * args.min_line_length / 100
    threshold_line_size = img.shape[0] * args.threshold_line_size / 100

    red = np.copy(img)
    red[:,:,1] = np.zeros([red.shape[0], red.shape[1]])
    red[:,:,0] = np.zeros([red.shape[0], red.shape[1]])
    gray = cv2.cvtColor(red, cv2.COLOR_BGR2GRAY)
    (thresh, bw) = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
    min_line = min_line_length
    red_lines = recognize(bw, min_line)
    while red_lines is None and min_line > 0:
        min_line = min_line - 50
        red_lines = recognize(bw, min_line)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, bw) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    min_line = min_line_length
    lines = recognize(bw, min_line)
    while lines is None and min_line > 0:
        min_line = min_line - 50
        lines = recognize(gray, min_line)

    avg_lines = []

    if red_lines is not None:
        tmp = average_vector(red_lines, 'laser', threshold_line_size)
        avg_lines.extend(tmp)

    if lines is not None:
        tmp = average_vector(lines, 'guide', threshold_line_size)
        avg_lines.extend(tmp)

    if args.dump:
        dump_image(img, red_lines, lines, avg_lines)

    print(json.dumps(avg_lines))

def recognize(img, min_line_length):
    kernel_size = 5
    blur = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    low_threshold = 50
    high_threshold = 250
    edges = cv2.Canny(blur, low_threshold, high_threshold)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    max_line_gap = 25  # maximum gap in pixels between connectable line segments

    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                    min_line_length, max_line_gap)

    return lines

def average_vector(lines, type, threshold_line_size):
    if lines is None:
        return None

    array = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            # m = (y2 - y1) / (x2 - x1)
            # y = 0
            # x = int(((y2 - y1) / m) + x1)
            # ye = 600
            # xe = int(((ye - y) / m) + x)
            # array.append([x, y, xe, ye])
            y = 0
            x = int(x1 + ((y - y1) * (x2 -x1)/(y2 - y1)))
            array.append([x, y, x1, y1])

    array = np.array(array, dtype='i4')
    array = np.sort(array.view('i4,i4,i4,i4'), order=['f0'], axis=0)

    tmp = {'x1':0,'y1':0,'x2':0,'y2':0,'type':type}
    length = 0
    threshold = array[0][0][0] + threshold_line_size
    listing = []
    v_min = []
    v_max = []
    for line in array:
        for x1, y1, x2, y2 in line:
            if x1 > threshold:
                if len(v_max) == 0:
                    v_max = v_min
                tmp['x1'] = int((v_max[0] + v_min[0]) / 2)
                tmp['y1'] = int((v_max[1] + v_min[1]) / 2)
                tmp['x2'] = int((v_max[2] + v_min[2]) / 2)
                tmp['y2'] = int((v_max[3] + v_min[3]) / 2)
                listing.append(tmp)

                tmp = {'x1':0,'y1':0,'x2':0,'y2':0,'type':type}
                threshold = x1 + threshold_line_size
            if len(v_min) == 0:
                v_min = [x1, y1, x2, y2]
            else:
                v_max = [x1, y1, x2, y2]

    if len(v_max) == 0:
        v_max = v_min
    tmp['x1'] = int((v_max[0] + v_min[0]) / 2)
    tmp['y1'] = int((v_max[1] + v_min[1]) / 2)
    tmp['x2'] = int((v_max[2] + v_min[2]) / 2)
    tmp['y2'] = int((v_max[3] + v_min[3]) / 2)
    listing.append(tmp)

    return listing

process()
