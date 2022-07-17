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
show_grid = False
show_crosshair = False
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
    
    text2 = my_font.render('C:Crosshair  B:BoundingBox  G:Gridlines', False, text_color)
    text_rect2 = text2.get_rect(center=(14*PIXEL_SIZE, 29*PIXEL_SIZE + 10))
    disp.blit(text2, text_rect2)

def draw_gridlines(disp, color=(100,100,100)):
    if show_grid:
        for i in range(29):
            #vertical lines
            pygame.draw.line(disp, color, (i*PIXEL_SIZE, 0), (i*PIXEL_SIZE, 28*PIXEL_SIZE), 1)
            #horizontal lines
            pygame.draw.line(disp, color, (0, i*PIXEL_SIZE), (28*PIXEL_SIZE, i*PIXEL_SIZE), 1)
        
    #center box
    if show_crosshair:
        mid = 28*PIXEL_SIZE // 2 - 5
        pygame.draw.rect(disp, (0, 255, 0), (mid, mid, 10, 10))

def draw_bb(disp):
    if not show_boundingbox:
        return
    t = b = l = r = 0
    for i, row in enumerate(pixels):
        if sum([x[0] for x in row]) > 0:
            l = i
            break
        
    for i, row in enumerate(pixels[::-1]):
        if sum([x[0] for x in row]) > 0:
            r = 28 - i
            break
        
    pixels_T = [[pixels[j][i] for j in range(len(pixels))] for i in range(len(pixels[0]))]
    
    for i, row in enumerate(pixels_T):
        if sum([x[0] for x in row]) > 0:
            t = i
            break
        
    for i, row in enumerate(pixels_T[::-1]):
        if sum([x[0] for x in row]) > 0:
            b = 28 - i
            break
    
    color = (0, 0, 255)
    pygame.draw.line(disp, color, (l*PIXEL_SIZE, t*PIXEL_SIZE), (l*PIXEL_SIZE, b*PIXEL_SIZE), 1) #TOP
    pygame.draw.line(disp, color, (r*PIXEL_SIZE, t*PIXEL_SIZE), (r*PIXEL_SIZE, b*PIXEL_SIZE), 1) #TOP
    pygame.draw.line(disp, color, (l*PIXEL_SIZE, t*PIXEL_SIZE), (r*PIXEL_SIZE, t*PIXEL_SIZE), 1) #TOP
    pygame.draw.line(disp, color, (l*PIXEL_SIZE, b*PIXEL_SIZE), (r*PIXEL_SIZE, b*PIXEL_SIZE), 1) #TOP
    
def fill_px(loc, w=1):
    x, y = loc
    pixels[x][y] = (255, 255, 255) if w == 1 else (0, 0, 0)
    for i in range(28):
        for j in range(28):
            if (x-i)**2 + (y-j)**2 <= brush_size**2:
                dist = math.sqrt((x-i)**2 + (y-j)**2)
                old_b = pixels[i][j][0]
                brightness = max(min(255, w*int((brush_size - dist) * 255 / brush_size) + old_b), 0)
                pixels[i][j] = (brightness, brightness, brightness)
    old_pixel = (x, y)
    
def draw_pixels(disp):
    for i, row in enumerate(pixels):
        for j, color in enumerate(row):
            pygame.draw.rect(disp, color, (i*PIXEL_SIZE, j*PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE))

def make_prediction(model, x):
    z = model.predict(x)
    yhat = tf.nn.softmax(z).numpy()
    return yhat

def img_to_x():
    data = []
    for i in range(28):
        for j in range(28):
            data.append(pixels[j][i][0] / 255)
            
    return np.array(data)   
        
def draw_probs(disp, yhat_in, color=(100,100,100)):
    yhat = yhat_in[0]
    pred = yhat.argmax()
    
    #check if image is blank
    if is_blank():
        yhat = np.zeros(10)
        pred = -1
    
    my_font = pygame.font.SysFont('mono', 15, bold=True)
    
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

print('\n\n-->Loading model...')
application_path = os.path.dirname(sys.executable)
model = tf.keras.models.load_model('./best_0-11_2-38')

disp = pygame.display.set_mode((28*PIXEL_SIZE, 34*PIXEL_SIZE))
pygame.display.set_caption('Digit Recognition')

run = True
clock = pygame.time.Clock()
frames_to_predict = 0
while run:
    clock.tick(120) #FPS
    
    draw_bg(disp)
    
    #model
    if frames_to_predict == 0:
        yhat = make_prediction(model, img_to_x().reshape(1,-1))
        certainity = yhat[0, yhat.argmax(axis=1)]
        print(yhat)
        
        frames_to_predict = 30 #runs every n frames
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
                
            if event.key == pygame.K_c:
                show_crosshair = not show_crosshair
                
            if event.key == pygame.K_b:
                show_boundingbox = not show_boundingbox
        
        if event.type == pygame.QUIT: #pressed the 'X'
            run = False