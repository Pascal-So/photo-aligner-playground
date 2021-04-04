import cv2
import math
import numpy as np
import sys

def show_image(img):
    cv2.imshow("", img)
    cv2.waitKey(0)

def rotate_image(img, angle):
    h, w = img.shape[:2]
    m = max(w, h)
    matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle * 180 / math.pi, 1.0)
    matrix[0][2] += (m - w) / 2
    matrix[1][2] += (m - h) / 2

    return cv2.warpAffine(img, matrix, (m, m))

def scale_image(img, scale):
    h, w = img.shape[:2]
    dim = (int(w * scale), int(h * scale))

    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

def draw_lines(img, lines, colour=(0,255,0)):
    ''' Modifies the image inplace. '''

    for i, [[x1,y1,x2,y2]] in enumerate(lines):
        cv2.line(img, (x1,y1), (x2,y2), colour, 2)


def edge_detect(img):
    ''' Get a list of edges in the image. '''

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 500, 1500, apertureSize = 5)
    lines = cv2.HoughLinesP(image=edges, rho=3, theta=np.pi/500, threshold=200, minLineLength=100, maxLineGap=6)
    if lines is None:
        return np.array([])
    else:
        return lines.reshape((-1, 4))

def prevalent_angle(lines, nr_buckets=60):
    ''' Returns the most common direction of the lines modulo pi/2.

    Todo: weighted by length

    '''

    buckets = np.zeros(nr_buckets)

    for [x1,y1,x2,y2] in lines:
        dx = x2 - x1
        dy = y2 - y1
        angle = (math.atan2(dy, dx) + math.pi * 2) % (math.pi / 2)
        buckets[math.floor(angle * nr_buckets / (math.pi / 2))] += 1

    angle = (buckets.argmax() + 0.5) * math.pi / 2 / nr_buckets
    if angle < math.pi / 4:
        return angle
    else:
        return angle - math.pi / 2

def fix_angle(img):
    angle = prevalent_angle(edge_detect(img))
    return rotate_image(img, angle)

def match_scale_and_crop(img, template):
    best = (0,0,(0,0))

    for scale in np.geomspace(0.95, 0.6, 20):
        resized = scale_image(img, scale)
        result = cv2.matchTemplate(resized, template, cv2.TM_CCOEFF_NORMED)
        np.nan_to_num(result, copy=False, nan=0.)
        score = np.max(result)
        index = np.unravel_index(np.argmax(result), result.shape)

        # print('scale: {} \tscore: {}'.format(scale, score))

        if score > best[0]:
            best = (score, scale, index)

    (score, scale, (y, x)) = best
    resized = scale_image(img, scale)

    [x1, y1, x2, y2] = rectangle_coords(x, y)
    cropped = resized[y1:y2, x1:x2]

    # draw_rectangle = resized.copy()
    # cv2.rectangle(draw_rectangle, (x1, y1), (x2, y2), (255,0,0), 2)
    # show_image(draw_rectangle)
    # sys.exit()

    return cropped



# Change template configuration here:

def load_template():
    template_path = 'img/template_vwr.png'

    return cv2.imread(template_path)

def rectangle_coords(x, y):
    ''' Get rectangle coordinates

    Given the x and y coordinates of the matched template location, return
    the desired bounding box coordinates of the final image.

    Order: [x1, y1, x2, y2]

    '''

    return [x-60, y+40, x+180, y+140]

if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit('Usage: {} input-path output-path'.format(sys.argv[0]))

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    angle_fixed = fix_angle(cv2.imread(input_path))

    template = load_template()
    output = match_scale_and_crop(angle_fixed, template)

    cv2.imwrite(output_path, output)
