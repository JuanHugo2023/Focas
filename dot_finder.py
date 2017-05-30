from PIL import Image
import numpy as np

def get_dots(filename=None):
    if (filename==None):
        filename = raw_input('Input file name:')   

    seals_im = Image.open(filename)
    seals_ar = np.array(seals_im)

    print("Loaded {} which has size {}".format(filename, seals_ar.shape))
    dim0 = seals_ar.shape[0]
    dim1 = seals_ar.shape[1]

    #Construct boolean masks the size of the image which
    #are TRUE whenever the pixels might be part of a red dot
    #(or brown, or blue, or green)
    red_mask = ((seals_ar[:, :, 0] >= 220) & (seals_ar[:, :, 1] <= 30) & (seals_ar[:, :, 2] <= 30))
    brown_mask = ((seals_ar[:, :, 0] >= 70) & (seals_ar[:, :, 0] <= 100)
                  & (seals_ar[:, :, 1] >= 35) & (seals_ar[:, :, 1] <= 65)
                  & (seals_ar[:, :, 2] <= 30))
    blue_mask = ((seals_ar[:, :, 0] >= 10) & (seals_ar[:, :, 0] <= 45)
                 & (seals_ar[:, :, 1] >= 45) & (seals_ar[:, :, 1] <= 75)
                 & (seals_ar[:, :, 2] >= 160) & (seals_ar[:, :, 2] <= 215))
    green_mask = ((seals_ar[:, :, 0] >= 20) & (seals_ar[:, :, 0] <= 55)
                  & (seals_ar[:, :, 1] >= 160) & (seals_ar[:, :, 1] <= 190)
                  & (seals_ar[:, :, 2] <= 40))

    pink_mask = ((seals_ar[:, :, 0] >= 220) & (seals_ar[:, :, 0] <= 260)
                  & (seals_ar[:, :, 1] <= 40)
                  & (seals_ar[:, :, 2] >= 230) & (seals_ar[:, :, 2] <= 260))
    
    red_indices = np.where(red_mask)
    brown_indices = np.where(brown_mask)
    blue_indices = np.where(blue_mask)
    green_indices = np.where(green_mask)
    pink_indices = np.where(pink_mask)

    red_dot_coords = np.zeros((0, 2), dtype='int')
    brown_dot_coords = np.zeros((0, 2), dtype='int')
    blue_dot_coords = np.zeros((0, 2), dtype='int')
    green_dot_coords = np.zeros((0, 2), dtype='int')
    pink_dot_coords = np.zeros((0, 2), dtype='int')

    if (len(red_indices) > 0):
        for ind in range(len(red_indices[0])):
            i, j  = red_indices[0][ind], red_indices[1][ind]
            if ((red_mask[i, j]) and (delete_connected_component(red_mask, i, j) > 5)):
                red_dot_coords = np.append(red_dot_coords, [[i, j]], axis=0)
                
    if (len(brown_indices) > 0):
        for ind in range(len(brown_indices[0])):
            i, j  = brown_indices[0][ind], brown_indices[1][ind]
            if ((brown_mask[i, j]) and (delete_connected_component(brown_mask, i, j) > 8)):
                brown_dot_coords = np.append(brown_dot_coords, [[i, j]], axis=0)
                
                
    if (len(blue_indices) > 0):
        for ind in range(len(blue_indices[0])):
            i, j  = blue_indices[0][ind], blue_indices[1][ind]
            if ((blue_mask[i, j]) and (delete_connected_component(blue_mask, i, j) > 5)):
                blue_dot_coords = np.append(blue_dot_coords, [[i, j]], axis=0)
                
                
    if (len(green_indices) > 0):
        for ind in range(len(green_indices[0])):
            i, j  = green_indices[0][ind], green_indices[1][ind]
            if ((green_mask[i, j]) and (delete_connected_component(green_mask, i, j) > 5)):
                green_dot_coords = np.append(green_dot_coords, [[i, j]], axis=0)

    if (len(pink_indices) > 0):
        for ind in range(len(pink_indices[0])):
            i, j  = pink_indices[0][ind], pink_indices[1][ind]
            if ((pink_mask[i, j]) and (delete_connected_component(pink_mask, i, j) > 5)):
                pink_dot_coords = np.append(pink_dot_coords, [[i, j]], axis=0)
                
    
    d = {'red': red_dot_coords, 'brown': brown_dot_coords, 'blue': blue_dot_coords, 'green': green_dot_coords, 'pink': pink_dot_coords}
    print('Found {} red, {} brown, {} blue, {} green, {} pink'.format(len(d['red']), len(d['brown']), len(d['blue']), len(d['green']), len(d['pink'])))
    return d




def delete_connected_component(mask, row, col):
    mask[row, col] = False
    counter = 1
    if ((row < mask.shape[0] - 1) and (mask[row+1, col])):
        counter += delete_connected_component(mask, row+1, col)
    if ((row > 0) and (mask[row-1, col])):
        counter += delete_connected_component(mask, row-1, col)
    if ((col < mask.shape[1] - 1) and (mask[row, col+1])):
        counter += delete_connected_component(mask, row, col+1)
    if ((col > 0) and (mask[row, col-1])):
        counter += delete_connected_component(mask, row, col-1)
    return counter

def display_locations(coords, filename=None):
    if (filename==None):
        filename = raw_input('Input file name:')   

    print('attempting to open {}'.format(filename))
    image_ar = np.array(Image.open(filename))
    
    dim0 = image_ar.shape[0]
    dim1 = image_ar.shape[1]
    window_radius = 30
    mask = np.zeros((dim0, dim1, 3), dtype='bool')
    
    for ind in range(len(coords)):
        row_min = max(coords[ind][0] - window_radius, 0)
        row_max = min(coords[ind][0] + window_radius, dim0)
        col_min = max(coords[ind][1] - window_radius, 0)
        col_max = min(coords[ind][1] + window_radius, dim1)
        mask[row_min:row_max, col_min:col_max, :] = True

    Image.fromarray(mask * image_ar).show()
    
