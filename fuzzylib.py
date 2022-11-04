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

def f(i):
    return round(100 * i)
    # return float("{:.2f}".format(i))

def inv(i):
    return i/100

def prod(a, b):
    return a+b - a*b

def tri(x, color_intensity):
    scale = color_intensity/255
    if x <= 0.5:
        return scale * 2 * x
    else:
        return scale * 2 * (1-x)
    
def tri_min(x, color_intensity, minimum=0.1):
    peak = color_intensity/255
    add = abs(0.5-x)/0.5 * minimum
    return min(minimum + (peak - minimum) * (0.5 - np.abs(0.5-x)) * 2, peak)

class FuzzyColor:
    def __init__(self, rgb, fn=tri_min, conorm_fn=prod):
        # save original representation 
        r, g, b = rgb
        self._set_fn = fn
        self._conorm_fn = conorm_fn
        
        # initialize representation
        self._fuzzy_color = {}
        for i in range(0, 101):
            self._fuzzy_color[i] = 0.0
        
#         # red 
        red = {}
        for i in range(0, 101):
            red[i] = fn(inv(i), r)

#         # first range
        xs = red.keys()
        ys = [red[x] for x in xs]
        xs = [round(x * 0.66) - 33 for x in xs]
        red_left_part = dict(zip(xs, ys))

        for i in range(0, 34):
            self._fuzzy_color[i] = red_left_part[i]
        
        # second range
        xs = red.keys()
        ys = [red[x] for x in xs]
        xs = [round(x*0.66) + 66 for x in xs]
        red_right_part = dict(zip(xs, ys))
        for i in range(67, 101, 1):
            self._fuzzy_color[i] = red_right_part[i]
        
        # green
        green = {}
        for i in range(0, 101, 1):
            green[i] = fn(inv(i), g)
        
        xs = green.keys()
        ys = [green[x] for x in xs]
        xs = [round(x * 0.666) for x in xs]
        green_part = dict(zip(xs, ys))
        
        for i in range(0, 68, 1):
            self._fuzzy_color[i] = max(self._fuzzy_color.get(i) or 0, green_part[i])
            
        # blue 
        # XXX Quantization error in blue channel
        # FIXME use fidelity of 1000 to fix
        blue = {}
        for i in range(0, 101, 1):
            blue[i] = fn(inv(i), b)
            
        xs = blue.keys()
        ys = [blue[x] for x in xs]
    
        xs = [round(x * 0.67) + 33 for x in xs]
        blue_part = dict(zip(xs, ys))
        for i in range(33, 101, 1):
            self._fuzzy_color[i] = max(self._fuzzy_color.get(i) or 0, blue_part[i])
            
    def plot(self):
        show(self)
        
    def power(fc, n):
        xs = fc._fuzzy_color.keys()
        ys = [fc._fuzzy_color[c]**n for c in xs]
        _map = dict(zip(xs, ys))
        r, g, b = FuzzyColor.to_rgb(_map)
        return FuzzyColor((r, g, b), fc._set_fn, fc._conorm_fn)
            
    
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
        r, g, b = _map[0], _map[33], _map[66]
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
    

def fuzzy_denoise(image, set_fn=tri_min, conorm_fn=prod, power_fn=FuzzyColor.power):
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
                
            if 0 <= y+1 < fuzzy_input.shape[0] and 0 <= x+1 < fuzzy_input.shape[1]:
                n = fuzzy_input[y+1][x+1]
                neighbours.append(n)
            
            if 0 <= y-1 < fuzzy_input.shape[0] and 0 <= x-1 < fuzzy_input.shape[1]:
                n = fuzzy_input[y-1][x-1]
                neighbours.append(n)
                
            if 0 <= y+1 < fuzzy_input.shape[0] and 0 <= x-1 < fuzzy_input.shape[1]:
                n = fuzzy_input[y+1][x-1]
                neighbours.append(n)
            
            
            if 0 <= y-1 < fuzzy_input.shape[0] and 0 <= x+1 < fuzzy_input.shape[1]:
                n = fuzzy_input[y-1][x+1]
                neighbours.append(n)


            if len(neighbours) > 0:
                result = neighbours[0]
                for i in range(1, len(neighbours)):
                    result += neighbours[i]

                averaged = power_fn(result, len(neighbours))
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

def dark_noise_fn(img, noise_level, noise_max_intensity):
    noise = np.random.random(size=(img.shape[0], img.shape[1]))
    mask = noise < noise_level
    noisy = img.copy()
    noisy[mask] = (np.random.rand(mask.sum(), 3) * noise_max_intensity).astype(int)
    return noisy

def salt_and_pepper_noise(img, noise_level):
    noise = np.random.random(size=(img.shape[0], img.shape[1]))
    mask = noise < noise_level
    noisy = img.copy()

    rnd_values = (np.random.rand(mask.sum()) > 0.5) * 255
    noisy[mask, 0] = noisy[mask, 1] = noisy[mask, 2] = rnd_values 
    return noisy

def rnd_noise(img, noise_level):
    noise = np.random.random(size=(img.shape[0], img.shape[1]))
    mask = noise < noise_level
    noisy = img.copy()

    rnd_values = np.random.rand(mask.sum(), 3) * 255
    noisy[mask] = rnd_values
    return noisy

def conorm_max(a, b):
    return min(1, max(a, b))
    
def conorm_avg(a, b):
    return (a+b)/2

def conorm_weighted_avg(a, b, gravity=0.1):
    coef = (1-gravity)/2
    return (coef*a + coef*b + gravity*0.5)

def power_minus(fc, n):
    return FuzzyColor.power(fc, max(0, n-1))

def power_div2(fc, n):
    return FuzzyColor.power(fc, max(0, round(n/2)))

def w2(a, b, lim=0.2):
    a1 = (a+b)/2
    if a1 <= lim:
        return a1*2
    
    if lim < a1 < (1-lim):
        return lim*2 + a1*(1-4*lim)
    
    if a1 >= (1-lim):
        return (1-2*lim) + (a1-(1-lim))*2

def nothing(fc, n):
    return fc
    # return FuzzyColor.power(fc, max(0, round(n/2)))

def test_and_compare(tests, max_h, set_fn, power_fn, conorm_fn, noise_fn):
    vstacks = []
    for t in tqdm(tests):
        # Read and resize image
        img = read_rgb(t)
        size_h = max_h
        coeff = img.shape[1]/img.shape[0]
        size_w = int(size_h * coeff)
        img = resize(img, size=(size_w, size_h))

        # Introduce noise
        noisy = noise_fn(img)
        
        # Denoise openCV 
        denoised_nlmeans = cv2.fastNlMeansDenoisingColored(noisy,None,20,20,7,21) 

        # Denoise gaussian (blur)
        denoised_blur = blur(noisy)
        
        # Denoise (crisp)
        denoised_crisp = crisp_denoise(noisy)
        
        # Denoised with fuzzy logic
        denoised_fuzzy = fuzzy_denoise(noisy, set_fn=set_fn, power_fn=power_fn, conorm_fn=conorm_fn)
        vstack = np.vstack([img, noisy, denoised_nlmeans, denoised_blur, denoised_crisp, denoised_fuzzy])
        vstacks.append(vstack)

    grid = np.hstack(vstacks)
    return grid