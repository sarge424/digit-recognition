import math
import numpy as np

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

import pygame
pygame.init()
pygame.font.init()

#GLOBALS
PIXEL_SIZE = 20
FPS = 200
MIN_AREA = 10
show_grid = False
show_boundingbox = False
old_pixel = (-1, -1)
brush_size = 1.5

pixels = [[(0,0,0) for j in range(28)] for i in range(28)]

def draw_bg(disp):
    pygame.draw.rect(disp, (0, 0, 0), (0, 0, 28*PIXEL_SIZE, 28*PIXEL_SIZE)) # bgcolor
    pygame.draw.rect(disp, (100, 100, 100), (0, 30*PIXEL_SIZE, 28*PIXEL_SIZE, 4*PIXEL_SIZE)) # bgcolor
    
    pygame.draw.line(disp, (255,255,255), (0, 30*PIXEL_SIZE), (28*PIXEL_SIZE, 30*PIXEL_SIZE), 2)
    pygame.draw.line(disp, (255,255,255), (0, 28*PIXEL_SIZE), (28*PIXEL_SIZE, 28*PIXEL_SIZE), 2)
    
    my_font = pygame.font.SysFont('mono', 20, bold=True)
    text_color = (255, 255, 255)
    text = my_font.render('L-Click:Draw  R-Click:Erase  M-Click:Clear', False, text_color)
    text_rect = text.get_rect(center=(14*PIXEL_SIZE, 28*PIXEL_SIZE + 12))
    disp.blit(text, text_rect)
    
    text2 = my_font.render('B:Toggle Bounding Box    G:Toggle Gridlines', False, text_color)
    text_rect2 = text2.get_rect(center=(14*PIXEL_SIZE, 29*PIXEL_SIZE + 10))
    disp.blit(text2, text_rect2)

def draw_gridlines(disp, color=(100,100,100)):
    if show_grid:
        for i in range(29):
            #vertical lines
            pygame.draw.line(disp, color, (i*PIXEL_SIZE, 0), (i*PIXEL_SIZE, 28*PIXEL_SIZE), 1)
            #horizontal lines
            pygame.draw.line(disp, color, (0, i*PIXEL_SIZE), (28*PIXEL_SIZE, i*PIXEL_SIZE), 1)

def calc_bb():
    t = b = l = r = 0
    for i, row in enumerate(pixels):
        if sum([x[0] for x in row]) > 0:
            t = i
            break
        
    for i, row in enumerate(pixels[::-1]):
        if sum([x[0] for x in row]) > 0:
            b = 28 - i
            break
        
    pixels_T = [[pixels[j][i] for j in range(len(pixels))] for i in range(len(pixels[0]))]
    
    for i, row in enumerate(pixels_T):
        if sum([x[0] for x in row]) > 0:
            l = i
            break
        
    for i, row in enumerate(pixels_T[::-1]):
        if sum([x[0] for x in row]) > 0:
            r = 28 - i
            break
        
    return t,b,l,r

def calc_area():
    t, b, l, r = calc_bb()
    area = (t-b) * (l-r) * 100 / 28**2
    return round(area, 2)

def draw_bb(disp):
    if not show_boundingbox:
        return
    t, b, l, r = calc_bb()
    color = (0, 0, 255)
    pygame.draw.line(disp, color, (l*PIXEL_SIZE, t*PIXEL_SIZE), (l*PIXEL_SIZE, b*PIXEL_SIZE), 1) #TOP
    pygame.draw.line(disp, color, (r*PIXEL_SIZE, t*PIXEL_SIZE), (r*PIXEL_SIZE, b*PIXEL_SIZE), 1) #TOP
    pygame.draw.line(disp, color, (l*PIXEL_SIZE, t*PIXEL_SIZE), (r*PIXEL_SIZE, t*PIXEL_SIZE), 1) #TOP
    pygame.draw.line(disp, color, (l*PIXEL_SIZE, b*PIXEL_SIZE), (r*PIXEL_SIZE, b*PIXEL_SIZE), 1) #TOP
    
def fill_px(loc, w=1):
    x, y = loc
    pixels[y][x] = (255, 255, 255) if w == 1 else (0, 0, 0)
    for i in range(28):
        for j in range(28):
            if (x-j)**2 + (y-i)**2 <= brush_size**2:
                dist = math.sqrt((x-j)**2 + (y-i)**2)
                old_b = pixels[i][j][0]
                brightness = max(min(255, w*int((brush_size - dist) * 255 / brush_size) + old_b), 0)
                pixels[i][j] = (brightness, brightness, brightness)
    old_pixel = (x, y)
    
def draw_pixels(disp):
    for i, row in enumerate(pixels):
        for j, color in enumerate(row):
            pygame.draw.rect(disp, color, (j*PIXEL_SIZE, i*PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE))

def make_prediction(model, x):
    z = model.predict(x)
    yhat = tf.nn.softmax(z).numpy()
    return yhat

def img_to_x():
    data = []
    p_center = center_img()
    for i in range(28):
        for j in range(28):
            data.append(p_center[i][j][0] / 255)
            
    return np.array(data)   
        
def center_img():
    t, b, l, r = calc_bb()
    
    if b-t == 0 or l-r == 0:
        return pixels
    
    pixels_sub = [[pixels[i][j] for j in range(l, r)] for i in range(t, b)]
    
    r = len(pixels_sub)
    c = len(pixels_sub[0])
    r_pad = (28 - r) / 2
    c_pad = (28 - c) / 2
    
    pixels_c = [[(0,0,0)] * math.floor(c_pad) + pixels_sub[i] + [(0,0,0)] * math.ceil(c_pad) for i in range(r)]
    pixels_centered = [[(0,0,0) for _ in range(28)]] * math.floor(r_pad) + pixels_c + [[(0,0,0) for _ in range(28)]] * math.ceil(r_pad)
    
    return pixels_centered
        
def px_debug(pixels_in):
    print('debug:', len(pixels_in), len(pixels_in[0]))
    for row in pixels_in:
        print('|', end='')
        for col in row:
            if col[0] > 0:
                print('##', end='')
            else:
                print('  ', end='')
        print('|\n', end='')

def draw_probs(disp, yhat_in, color=(100,100,100)):
    
    my_font = pygame.font.SysFont('mono', 15, bold=True)
    
    area = calc_area()
    if area < MIN_AREA:
        text = my_font.render(f'Bounding box is too small ({area}% area of minimum {MIN_AREA}% covered)', False, (255, 225, 0))
        text_rect = text.get_rect(center=(14*PIXEL_SIZE, 30*PIXEL_SIZE + 10))
        disp.blit(text, text_rect)
        return
        
    yhat = yhat_in[0]
    pred = yhat.argmax()
    
    box_size = 28*PIXEL_SIZE // 10
    y_height = 4*PIXEL_SIZE
    for n, y in enumerate(yhat):
        #probabilities
        color = (0, 255, 0) if n == pred else (255, 255, 255)
        y_h = y * y_height
        y_top = 30*PIXEL_SIZE + (y_height - y_h)
        pygame.draw.rect(disp, color, (n*box_size, y_top, box_size, y_h))
        
        #labels
        text_color = (0, 0, 0) if n == pred else (255, 255, 255)
        text = my_font.render(f'{n}:{round(yhat[n]*100):2.0f}', False, text_color)
        text_rect = text.get_rect(center=((n + 0.5) * box_size, 30*PIXEL_SIZE + 10))
        disp.blit(text, text_rect)
               
def is_blank():
    for row in pixels:
        for color in row:
            if not color == (0, 0, 0):
                return False
    return True

def resource_path(relative_path):
    """ Get the absolute path to the resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

print('\n\n-->Loading model...')
application_path = os.path.dirname(sys.executable)
model = tf.keras.models.load_model(resource_path('./model'))

disp = pygame.display.set_mode((28*PIXEL_SIZE, 34*PIXEL_SIZE))
pygame.display.set_caption('Digit Recognition')

run = True
clock = pygame.time.Clock()
frames_to_predict = 0
yhat = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
while run:
    clock.tick(FPS) #FPS
    
    draw_bg(disp)
    
    #model
    if frames_to_predict <= 0 and calc_area() >= MIN_AREA:
        yhat = make_prediction(model, img_to_x().reshape(1,-1))
        certainity = yhat[0, yhat.argmax(axis=1)]    
        
        frames_to_predict = FPS // 4 #runs 4 times a sec
    else:
        frames_to_predict -= 1
    
    draw_pixels(disp)
    draw_gridlines(disp)
    draw_bb(disp)
    draw_probs(disp, yhat)
    
    pygame.display.flip() #updates the display
    for event in pygame.event.get():
        if pygame.mouse.get_pressed()[0]:
            m_pos = pygame.mouse.get_pos()
            m_x, m_y = m_pos
            m_pixel = (m_x // PIXEL_SIZE, m_y // PIXEL_SIZE) #get the pixel coords
            if m_pixel != old_pixel and m_y < 560: #make sure we're not in the toolbar
                fill_px(m_pixel, 1)
            
        if pygame.mouse.get_pressed()[1]:
            pixels = [[(0,0,0) for j in range(28)] for i in range(28)]
        
        if pygame.mouse.get_pressed()[2]:
            m_pos = pygame.mouse.get_pos()
            m_x, m_y = m_pos
            m_pixel = (m_x // PIXEL_SIZE, m_y // PIXEL_SIZE) #get the pixel coords
            if m_y < 560: #make sure we're not in the toolbar
                fill_px(m_pixel, -1)
        
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_g:
                show_grid = not show_grid
                
            if event.key == pygame.K_b:
                show_boundingbox = not show_boundingbox
                
            if event.key == pygame.K_d:
                center_img()
        
        if event.type == pygame.QUIT: #pressed the 'X'
            run = False