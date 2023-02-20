import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pathlib
from tqdm import tqdm
from skimage.measure import regionprops, label

text_images = [plt.imread(path) for path in pathlib.Path(".").glob("*.png")]


train_images = {}

for path in tqdm(sorted(pathlib.Path("./train").glob("*"))):
    symbol = path.name[-1]
    train_images[symbol] = []
    for image_path in sorted(path.glob("*.png")):
        symbol = path.name[-1]
        train_images[symbol].append(plt.imread(image_path))

def extract_features(image):
    if image.ndim == 3:
        gray = np.mean(image, 2)
        gray[gray > 0] = 1
        labeled = label(gray)
    else:
        labeled = image.astype("uint8")
    props = regionprops(labeled)[0]
    extent = props.extent
    eccentricity = props.eccentricity
    euler = props.euler_number
    rr, cc = props.centroid_local
    rr = rr / props.image.shape[0]
    cc = cc / props.image.shape[1]
    feret = (props.feret_diameter_max - 1) / (np.max(props.image.shape))
    return np.array([extent, eccentricity, euler, rr, cc, feret], dtype="f4")


knn = cv.ml.KNearest_create()

train = []
responses = []

sym2class = {symbol: i for i, symbol in enumerate(train_images)}
class2sym = {value: key for key, value in sym2class.items()}

for i, symbol in tqdm(enumerate(train_images)):
    for image in train_images[symbol]:
        train.append(extract_features(image))
        responses.append(sym2class[symbol])

train = np.array(train, dtype="f4")
responses = np.array(responses).reshape(-1, 1)

knn.train(train, cv.ml.ROW_SAMPLE, responses)

def image2text(image) -> str:
    gray = np.mean(image, 2)
    gray[gray > 0] = 1
    labeled = label(gray)
    regions = regionprops(labeled)
    answer = []
    region_x = {}
    for region in regions:
        region_x[region.bbox[1]] = region
    sorted_x = sorted(region_x)
    prev = 0
    space_min = 2**30
    space_max = 0
    for x in sorted_x:
        region = region_x[x]
        if region.bbox[1] - prev > space_max:
            space_max = region.bbox[1] - prev
        if region.bbox[1] - prev < space_min and region.bbox[1] - prev > 0:
            space_min = region.bbox[1] - prev
        prev = region.bbox[3]
    space_thresh = (space_max + space_min) / 2
    prev = 0
    for x in sorted_x:
        region = region_x[x]
        if region.bbox[1] - prev < 0:
            answer[-1] = 'i'
            prev = region.bbox[3]
            continue
        if region.bbox[1] - prev > space_thresh:
            answer.append(' ')
        features = extract_features(region.image).reshape(1, -1)
        ret, results, neighbours, dist = knn.findNearest(features, 5)
        answer.append(class2sym[int(ret)])
        prev=region.bbox[3]
    return "".join(answer)

for image in text_images:
    print(image2text(image))
    # break