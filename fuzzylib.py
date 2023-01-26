import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

def read_rgb(path):
    x = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    return x

def resize(img, size=(100, 100)):
    return cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)

def blur(x, it=1):
    for _ in range(it):
        x = cv2.GaussianBlur(x, (3, 3), cv2.BORDER_DEFAULT)
    return x

def show_rgb(*img, figsize=(8, 3), title=''):
    plt.figure(figsize=figsize)
    for i in img:
        plt.imshow(i, cmap='gray')
        plt.title(title)
        plt.show()

# fuzzy-algo
def max_conorm(a, b):
    return max(a, b)

def f(i):
    return round(100 * i)
    # return float("{:.2f}".format(i))

def inv(i, scale=99):
    return i/scale

def prod(a, b):
    return a+b - a*b

def tri(x, color_intensity):
    scale = color_intensity/255
    if x <= 0.5:
        return scale * 2 * x
    else:
        return scale * 2 * (1-x)
    
def tri_min(x, color_intensity, minimum=0.0):
    peak = color_intensity/255
    add = abs(0.5-x)/0.5 * minimum
    return min(minimum + (peak - minimum) * (0.5 - np.abs(0.5-x)) * 2, peak)

class FuzzyColor:
    def __init__(self, rgb, fn=tri_min, conorm_fn=prod):
        # save original representation 
        r, g, b = rgb
        self._set_fn = fn
        self._conorm_fn = conorm_fn
        scale = 99
        
        # initialize representation
        self._fuzzy_color = {}
        for i in range(0, scale):
            self._fuzzy_color[i] = 0.0
        
#         # red 
        red = {}
        for i in range(0, scale):
            red[i] = fn(inv(i), r)

        xs = red.keys()
        ys = [red[x] for x in xs]
        xs = [round(x * 0.33) for x in xs]
        red_left_part = dict(zip(xs, ys))

        for i in range(0, 33):
            self._fuzzy_color[i] = red_left_part[i]
        
        # green
        green = {}
        for i in range(0, scale):
            green[i] = fn(inv(i), g)
        
        xs = green.keys()
        ys = [green[x] for x in xs]
        xs = [round(x * 0.334) + 33 for x in xs]
        green_part = dict(zip(xs, ys))

        for i in range(33, 66):
            self._fuzzy_color[i] = max(self._fuzzy_color.get(i) or 0, green_part[i])
            
        blue = {}
        for i in range(0, scale):
            blue[i] = fn(inv(i), b)
            
        xs = blue.keys()
        ys = [blue[x] for x in xs]
    
        xs = [round(x * 0.334) + 66 for x in xs]
        print(xs)
        blue_part = dict(zip(xs, ys))
        for i in range(66, scale):
            self._fuzzy_color[i] = max(self._fuzzy_color.get(i) or 0, blue_part[i])
            
    def plot(self):
        show(self)
        
    def power(self, n):
        xs = self._fuzzy_color.keys()
        ys = [self._fuzzy_color[c]**n for c in xs]
        _map = dict(zip(xs, ys))
        r, g, b = FuzzyColor.to_rgb(_map)
        return FuzzyColor((r, g, b), self._set_fn, self._conorm_fn)
            
    
    def __repr__(self):
        r, g, b = self.get_rgb()
        return str({
            'r': r,
            'g': g,
            'b': b
        })
    
    def get_rgb(self):
        r, g, b = FuzzyColor.to_rgb(self._fuzzy_color)
        return r, g, b
    
    def to_rgb(_map):
        r, g, b = _map[16], _map[49], _map[82]
        r = round(r*255)
        g = round(g*255)
        b = round(b*255)
        return r, g, b
    

    def __add__(self, other):
        assert self._set_fn == other._set_fn
        assert self._conorm_fn == other._conorm_fn
        
        k = self._fuzzy_color.keys()
        _map = {}
        for i in k:
            av = self._fuzzy_color[i]
            bv = other._fuzzy_color[i]
            cv = self._conorm_fn(av, bv)
            _map[i] = cv

        r, g, b = _map[0], _map[33], _map[66]
        r = round(r*255)
        g = round(g*255)
        b = round(b*255)
        return FuzzyColor((r, g, b), self._set_fn, self._conorm_fn)
        
        
def show(*fc, figsize=(8, 3)):
    plt.figure(figsize=figsize)
    for f in fc:
        xs = f._fuzzy_color.keys()
        ys = [f._fuzzy_color[x] for x in xs]
        plt.plot(np.array(list(xs))/100, ys)
    plt.legend(range(len(fc)))
    plt.xlim((0, 1))
    plt.ylim((0, 1))

class FuzzyImage:
    def __init__(self, rgb_image, set_fn=tri_min, conorm_fn=prod, fuzzy_image=None):
        if fuzzy_image is None:
            self.input = rgb_image
            self.set_fn = set_fn
            self.conorm_fn = conorm_fn

            from collections import defaultdict
            self.fuzzy_image = defaultdict(dict)
            for y in range(rgb_image.shape[0]):
                for x in range(rgb_image.shape[1]):
                    pixel = rgb_image[y][x]
                    fuzzyPixel = FuzzyColor(pixel, set_fn, conorm_fn)
                    self.fuzzy_image[y][x] = fuzzyPixel
        else:
            import copy
            self.input = fuzzy_image.input.copy()
            self.set_fn = fuzzy_image.set_fn
            self.conorm_fn = fuzzy_image.conorm_fn
            self.fuzzy_image = copy.deepcopy(fuzzy_image.fuzzy_image)
    
    def __getitem__(self, key):
        return self.fuzzy_image[key]

    def __setitem__(self, key, value):
        self.fuzzy_image
    
    @property
    def shape(self):
        return self.input.shape
    
    def get_rgb(self):
        output = self.input.copy()
        for y in range(output.shape[0]):
            for x in range(output.shape[1]):
                output[y][x] = self.fuzzy_image[y][x].get_rgb()
    
        return output
    
    def copy(self):
        return FuzzyImage(rgb_image=None, fuzzy_image=self)
    

def fuzzy_denoise(image, set_fn=tri_min, conorm_fn=prod):
    fuzzy_input = FuzzyImage(image, set_fn=set_fn, conorm_fn=conorm_fn)
    fuzzy_output = FuzzyImage(image, set_fn=set_fn, conorm_fn=conorm_fn)
    for y in range(fuzzy_input.shape[0]):
        for x in range(fuzzy_input.shape[1]):
            neighbours = []
            if 0 <= y+1 < fuzzy_input.shape[0]:
                n = fuzzy_input[y+1][x]
                neighbours.append(n)

            if 0 <= y-1 < fuzzy_input.shape[0]:
                n = fuzzy_input[y-1][x]
                neighbours.append(n)

            if 0 <= x+1 < fuzzy_input.shape[1]:
                n = fuzzy_input[y][x+1]
                neighbours.append(n)

            if 0 <= x-1 < fuzzy_input.shape[1]:
                n = fuzzy_input[y][x-1]
                neighbours.append(n)

            if len(neighbours) > 0:
                result = neighbours[0]
                for i in range(1, len(neighbours)):
                    result += neighbours[i]

                averaged = result.power(len(neighbours))
                fuzzy_output[y][x] = averaged
                  
    return fuzzy_output.get_rgb()

def crisp_denoise(image):
    output = image.copy()
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            neighbours = []
            if 0 <= y+1 < image.shape[0]:
                n = image[y+1][x]
                neighbours.append(n)

            if 0 <= y-1 < image.shape[0]:
                n = image[y-1][x]
                neighbours.append(n)

            if 0 <= x+1 < image.shape[1]:
                n = image[y][x+1]
                neighbours.append(n)

            if 0 <= x-1 < image.shape[1]:
                n = image[y][x-1]
                neighbours.append(n)

            if len(neighbours) > 0:
                neigh = np.stack(neighbours)
                r_avg = neigh[:, 0].mean()
                g_avg = neigh[:, 1].mean()
                b_avg = neigh[:, 2].mean()

                output[y][x] = (r_avg, g_avg, b_avg)
                  
    return output
def test_and_compare(tests, max_h, noise_level, conorm_fn, noise_max_intensity=63):
    vstacks = []
    for t in tqdm(tests):
        # Read and resize image
        img = read_rgb(t)
        size_h = max_h
        coeff = img.shape[1]/img.shape[0]
        size_w = int(size_h * coeff)
        img = resize(img, size=(size_w, size_h))

        # Introduce noise
        
        noise = np.random.random(size=(img.shape[0], img.shape[1]))
        mask = noise < noise_level
        noisy = img.copy()
        
        noisy[mask] = (np.random.rand(mask.sum(), 3) * noise_max_intensity).astype(int)

        # Denoise openCV 
        denoised_nlmeans = cv2.fastNlMeansDenoisingColored(noisy,None,20,20,7,21) 

        # Denoise gaussian (blur)
        denoised_blur = blur(noisy)
        
        # Denoise (crisp)
        denoised_crisp = crisp_denoise(noisy)
        
        # Denoised with fuzzy logic
        denoised_fuzzy = fuzzy_denoise(noisy, conorm_fn=conorm_fn)
        vstack = np.vstack([img, noisy, denoised_nlmeans, denoised_blur, denoised_crisp, denoised_fuzzy])
        vstacks.append(vstack)

    grid = np.hstack(vstacks)
    return grid